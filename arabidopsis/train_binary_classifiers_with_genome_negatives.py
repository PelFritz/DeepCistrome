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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from pybedtools import BedTool
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

        for chrom, start, end, target in batch_x:
            seq = one_hot_encode(self.genome[chrom][int(start):int(end)])
            if seq.shape[0] == self.window_size:
                x.append(seq)
                y.append(target)

        x, y = np.array(x), np.array(y)
        return x, y


def create_pos_bins(tf_family, chroms):
    df = pd.read_csv(filepath_or_buffer=f"data/peaks/{tf_family}.bed", sep='\t', header=None,
                     dtype={0: str, 1: int, 2: int})
    df = df[df[0].isin(chroms)]

    peaks = []
    for c, s, e in df.values:
        midpoint = (s+e)//2
        peaks.append([c, max(0, midpoint-125), midpoint+125])
    peaks = pd.DataFrame(peaks)
    return peaks


def create_negative_bins(chrom_sizes, pos_bins, chroms, window_size=250, step_size=50):
    windows = BedTool().window_maker(g=chrom_sizes, w=window_size, s=step_size)
    windows = windows.filter(lambda r: r.chrom in chroms and r.stop - r.start == window_size).saveas('data/neg_bins.bed')
    pos_bed = BedTool().from_dataframe(pos_bins)
    neg_bed = windows.intersect(pos_bed, v=True)
    return neg_bed.to_dataframe()


def create_bins_list(tf_family, chroms):
    genome_size = pd.read_csv(filepath_or_buffer='data/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.fai',
                              sep='\t', dtype={0: str, 1: int, 2: int}, header=None)
    genome_size = genome_size[genome_size[0].isin(chroms)]
    genome_size.to_csv('data/genome/chrom_sizes.bed', sep='\t', index=False, header=False)

    pos_bins = create_pos_bins(tf_family, chroms)
    pos_bins = pos_bins[pos_bins[0].isin(chroms)]
    pos_bins[3] = [1]*pos_bins.shape[0]
    print(f'Number of positive bins {pos_bins.shape[0]}')

    neg_bins = create_negative_bins(chrom_sizes='data/genome/chrom_sizes.bed',
                                    chroms=chroms, pos_bins=pos_bins, window_size=250, step_size=50)
    neg_bins = neg_bins.sample(n=pos_bins.shape[0], random_state=42)
    neg_bins[3] = [0]*neg_bins.shape[0]
    neg_bins.columns = [0, 1, 2, 3]

    all_bins = pd.concat([pos_bins, neg_bins])
    all_bins = all_bins.sample(frac=1, random_state=42)
    all_bins[0] = all_bins[0].astype(str)
    print(all_bins.head())
    print(all_bins[0].unique())
    print(all_bins.shape)
    os.system('rm -rf data/neg_bins.bed')
    return all_bins.values.tolist()


for peaks_file in sorted(os.listdir('data/peaks')):
    tf_fam = peaks_file.replace('.bed', '')
    print(f'Training on: {tf_fam}')
    for chrom in chromosomes:
        print(f'Training on: {tf_fam}, chromosome: {chrom}')
        train_chroms = [i for i in chromosomes if i != chrom]
        valid_chrom = [chrom]
        train_dg = InputGenerator(genome=genome_path,
                                  bins_list=create_bins_list(tf_family=tf_fam, chroms=train_chroms),
                                  batch_size=32, window_size=250)
        valid_dg = InputGenerator(genome=genome_path,
                                  bins_list=create_bins_list(tf_family=tf_fam, chroms=valid_chrom),
                                  batch_size=32, window_size=250)

        model = compile_model(window_size=250)

        model_save_name = f"saved_models/{tf_fam}_chrom_{chrom}_gn.h5"
        cvs_log_save_name = f"results/{tf_fam}_{chrom}_gn.log"
        model.fit(train_dg, validation_data=valid_dg, epochs=100,
                  callbacks=[
                      ModelCheckpoint(filepath=model_save_name, monitor='val_loss', save_best_only=True, verbose=1),
                      EarlyStopping(monitor='val_loss', patience=5),
                      CSVLogger(cvs_log_save_name)
                   ]
                  )
