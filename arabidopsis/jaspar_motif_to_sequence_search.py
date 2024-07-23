#https://github.com/biopython/biopython/blob/master/Doc/Tutorial/chapter_motifs.rst
import pandas as pd
from pyjaspar import jaspardb
from Bio.Seq import Seq
import os
from pyfaidx import Fasta
from concurrent.futures import ProcessPoolExecutor
jdb_obj = jaspardb(release='JASPAR2024')


def seq_search(motif, query_seq):
    # Using the query sequence to create the background
    background = {letter: query_seq.count(letter)/len(query_seq)for letter in 'ACGT'}
    #print(background)
    # Pseudo_count to avoid -inf
    pwm = motif.counts.normalize(pseudocounts=0.1)
    pssm = pwm.log_odds()
    distribution = pssm.distribution(background=background, precision=10 ** 4)
    threshold = distribution.threshold_fpr(1e-5)
    hits = list(pssm.search(query_seq, threshold=threshold))
    #print(f'Consensus: {motif.consensus} and threshold: {threshold}')
    #print(hits)
    #for pos, score in hits:
    #    print(query_seq[pos:pos+len(motif)], pos, score)
    return 1 if len(hits) > 0 else 0


def get_num_hits(seq, family=None):
    hit_count = 0

    if family == 'CPP':
        for motif in jdb_obj.fetch_motifs(tf_class='CPP', tax_group='plants', collection=None):
            hit = seq_search(motif=motif, query_seq=seq)
            hit_count += hit
            if hit_count > 0:
                break

    elif family == 'TCP':
        for motif in jdb_obj.fetch_motifs(tf_class='TCP', tax_group='plants', collection=None):
            hit = seq_search(motif=motif, query_seq=seq)
            hit_count += hit
            if hit_count > 0:
                break

    elif family == 'C3H':
        for motif in jdb_obj.fetch_motifs(tf_class='C3H(C),C2HC zinc fingers factors', tax_group='plants',
                                          collection=None):
            hit = seq_search(motif=motif, query_seq=seq)
            hit_count += hit
            if hit_count > 0:
                break

    elif family == 'RWPRK':
        for motif in jdb_obj.fetch_motifs(tf_class='RWP-RK', tax_group='plants',
                                          collection=None):
            hit = seq_search(motif=motif, query_seq=seq)
            hit_count += hit
            if hit_count > 0:
                break

    elif family == 'BBR/BPC':
        for motif in jdb_obj.fetch_motifs(tf_class='BBR/BPC', tax_group='plants',
                                          collection=None):
            hit = seq_search(motif=motif, query_seq=seq)
            hit_count += hit
            if hit_count > 0:
                break

    elif family == 'AB13':
        for motif in jdb_obj.fetch_motifs(tf_name='ABI3', tax_group='plants',
                                          collection=None):
            hit = seq_search(motif=motif, query_seq=seq)
            hit_count += hit
            if hit_count > 0:
                break

    else:
        if jdb_obj.fetch_motifs(tf_family=family, tax_group='plants', collection=None):
            for motif in jdb_obj.fetch_motifs(tf_family=family, tax_group='plants', collection=None):
                hit = seq_search(motif=motif, query_seq=seq)
                hit_count += hit
                if hit_count > 0:
                    break

    return hit_count


family_to_jaspar_fam = {'ABI3VP1_tnt': 'AB13', 'AP2EREBP_tnt': 'ERF/DREB', 'ARF_ecoli': 'ARF',
                        'ARF_tnt': 'ARF', 'ARID_tnt': 'ARID', 'BBRBPC_tnt': 'BBR/BPC',
                        'BES1_tnt': 'BES/BZR', 'BZR_tnt': 'BES/BZR', 'C2C2YABBY_tnt': 'YABBY', 'C2C2dof_tnt': 'DOF',
                        'C2C2gata_tnt': 'C4-GATA-related', 'C2H2_tnt': 'C2H2', 'C3H_tnt': 'C3H', 'CAMTA_tnt': 'CAMTA',
                        'CPP_tnt': 'CPP', 'E2FDP_tnt': 'e2f', 'EIL_tnt': 'eil', 'FAR1_tnt': 'FRS/FRF',
                        'G2like_tnt': 'GARP_G2-like', 'GRF_tnt': 'grf', 'GeBP_tnt': 'gebp', 'HB_tnt': 'HD-ZIP',
                        'HSF_tnt': 'hsf', 'Homeobox_ecoli': 'HD-ZIP', 'Homeobox_tnt': 'HD-ZIP',
                        'LOBAS2_tnt': 'LBD', 'MADS_tnt': 'MIKC', 'MYB_tnt': 'MYB', 'MYBrelated_tnt': 'MYB-related',
                        'NAC_tnt': 'NAC', 'ND_tnt': 'nd', 'Orphan_tnt': 'orphan', 'RAV_tnt': 'rav', 'REM_tnt': 'rem',
                        'RWPRK_tnt': 'RWPRK', 'S1Falike_tnt': 's1falike', 'SBP_tnt': 'sbp',
                        'SRS_tnt': 'srs', 'TCP_tnt': 'TCP', 'Trihelix_tnt': 'Trihelix',
                        'WRKY_tnt': 'WRKY', 'ZFHD_tnt': 'PLINC', 'bHLH_tnt': 'bHLH', 'bZIP_tnt': 'bZIP',
                        'mTERF_tnt': 'mterf', 'zfGRF_tnt': 'zfgrf'}


def run_parallel(family_peaks):
    genome_file = 'data/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa'
    genome = Fasta(genome_file, as_raw=True, read_ahead=10000, sequence_always_upper=True)
    total_peaks, peaks_with_hits = 0, 0
    fam_name = family_peaks.replace('.bed', '')
    peaks = pd.read_csv(filepath_or_buffer=f'data/peaks/{family_peaks}', sep='\t', header=None,
                        dtype={0: str, 1: int, 2: int})
    print(f'Working on - {fam_name}: Number of Peaks - {peaks.shape[0]}')
    for chrom, start, end in peaks.values:
        seq = genome[chrom][start:end]
        num_hits = get_num_hits(seq=Seq(seq), family=family_to_jaspar_fam[fam_name])
        total_peaks += 1
        if num_hits > 0:
            peaks_with_hits += 1
    recall = peaks_with_hits / total_peaks
    print(f'{fam_name}: {recall}')

    return [fam_name, recall]


with ProcessPoolExecutor(max_workers=20) as executor:
    results = executor.map(run_parallel, sorted(os.listdir('data/peaks')))

results = pd.DataFrame(list(results), columns=['fam_name', 'recall'])
print(results.head(50))
results.to_csv(path_or_buf='results/jaspar_motif_to_sequence_search.csv', sep='\t', index=False)

#print(jdb_obj.fetch_motifs(tax_group='plants', collection=None))