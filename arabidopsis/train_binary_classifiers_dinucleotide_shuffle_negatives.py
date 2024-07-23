import pandas as pd
import numpy as np
from pyfaidx import Fasta
import os
import math
import tensorflow.keras.layers as kl
from tensorflow.keras.metrics import AUC
from tensorflow.keras import Model, optimizers
from tensorflow.keras.utils import Sequence
from utils import one_hot_encode
from deeplift.dinuc_shuffle import dinuc_shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
np.random.seed(42)

chromosomes = ['1', '2', '3', '4', '5']
genome_path = 'data/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa'


def compile_model(window_size):
    inputs = kl.Input(shape=(window_size, 4))
    x = kl.Conv1D(filters=256, kernel_size=21, padding='same')(inputs)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.MaxPooling1D(2)(x)

    conv_filters, conv_kernel_sizes = [60, 60, 120], [7, 7, 5]
    for n_filters, kernel_size in zip(conv_filters, conv_kernel_sizes):
        x = kl.Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

    x = kl.Flatten()(x)
    for units in [256, 256]:
        x = kl.Dense(units=units)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.4)(x)

    x = kl.Dense(units=1)(x)
    x = kl.Activation('sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='binary_crossentropy',
                  metrics=[AUC(multi_label=False, curve='ROC', name='auROC'),
                           AUC(multi_label=False, curve='PR', name='auPR'),
                           'accuracy'],
                  optimizer=optimizers.Adam(lr=0.002))
    model.summary()
    return model


class InputGenerator(Sequence):
    def __init__(self, genome, bins_list, batch_size, window_size):
        self.genome = Fasta(genome, as_raw=True, read_ahead=10000, sequence_always_upper=True)
        self.bins_list = bins_list
        self.batch_size = batch_size
        self.window_size = window_size

    def __len__(self):
        return math.ceil(len(self.bins_list) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.bins_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = []
        y = []

        for chrom, start, end in batch_x:
            seq_pos = one_hot_encode(self.genome[chrom][int(start):int(end)])
            seq_neg = dinuc_shuffle(one_hot_encode(self.genome[chrom][int(start):int(end)]),
                                    rng=np.random.default_rng(seed=42))
            if seq_pos.shape[0] == self.window_size:
                x.extend([seq_pos, seq_neg])
                y.extend([1, 0])

        x, y = np.array(x), np.array(y)
        return x, y


def create_pos_bins_list(tf_family, chroms):
    df = pd.read_csv(filepath_or_buffer=f"data/peaks/{tf_family}.bed", sep='\t', header=None,
                     dtype={0: str, 1: int, 2: int})
    df = df[df[0].isin(chroms)]

    peaks = []
    for c, s, e in df.values:
        midpoint = (s+e)//2
        peaks.append([c, max(0, midpoint-125), midpoint+125])
    peaks = pd.DataFrame(peaks)
    return peaks.values.tolist()


for peaks_file in sorted(os.listdir('data/peaks')):
    tf_fam = peaks_file.replace('.bed', '')
    print(f'Training on: {tf_fam}')
    for chrom in chromosomes:
        print(f'Training on: {tf_fam}, chromosome: {chrom}')
        train_chroms = [i for i in chromosomes if i != chrom]
        valid_chrom = [chrom]
        train_dg = InputGenerator(genome=genome_path,
                                  bins_list=create_pos_bins_list(tf_family=tf_fam, chroms=train_chroms),
                                  batch_size=32, window_size=250)
        valid_dg = InputGenerator(genome=genome_path,
                                  bins_list=create_pos_bins_list(tf_family=tf_fam, chroms=valid_chrom),
                                  batch_size=32, window_size=250)

        model = compile_model(window_size=250)

        model_save_name = f"saved_models/{tf_fam}_chrom_{chrom}.h5"
        cvs_log_save_name = f"results/{tf_fam}_{chrom}.log"
        model.fit(train_dg, validation_data=valid_dg, epochs=100,
                  callbacks=[
                      ModelCheckpoint(filepath=model_save_name, monitor='val_loss', save_best_only=True, verbose=1),
                      EarlyStopping(monitor='val_loss', patience=5),
                      CSVLogger(cvs_log_save_name)]
                  )
