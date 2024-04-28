from enum import Enum, unique


@unique
class DatasetRegistry(Enum):
    COLUMBIA = "columbia"
    COLUMBIA_OSN = "columbia_osn"
    COLUMBIA_WEBP = "columbia_webp"
    COVERAGE = "coverage"
    REALISTIC_TAMPERING = "realistic_tampering"
    REALISTIC_TAMPERING_WEBP = "realistic_tampering_webp"
    DSO1 = "dso1"
    DSO1_OSN = "dso1_osn"
    CASIA1_COPY_MOVE = "casia1_copy_move"
    CASIA1_SPLICING = "casia1_splicing"
    CASIA1_COPY_MOVE_OSN = "casia1_copy_move_osn"
    CASIA1_SPLICING_OSN = "casia1_splicing_osn"
    AUTOSPLICE100 = "autosplice_100"
    AUTOSPLICE90 = "autosplice_90"
    AUTOSPLICE75 = "autosplice_75"
    TRACE_NOISE_EXO = "trace_noise_exo"
    TRACE_NOISE_ENDO = "trace_noise_endo"
    TRACE_CFA_ALG_EXO = "trace_cfa_alg_exo"
    TRACE_CFA_ALG_ENDO = "trace_cfa_alg_endo"
    TRACE_CFA_GRID_EXO = "trace_cfa_grid_exo"
    TRACE_CFA_GRID_ENDO = "trace_cfa_grid_endo"
    TRACE_JPEG_GRID_EXO = "trace_jpeg_grid_exo"
    TRACE_JPEG_GRID_ENDO = "trace_jpeg_grid_endo"
    TRACE_JPEG_QUALITY_EXO = "trace_jpeg_quality_exo"
    TRACE_JPEG_QUALITY_ENDO = "trace_jpeg_quality_endo"
    TRACE_HYBRID_EXO = "trace_hybrid_exo"
    TRACE_HYBRID_ENDO = "trace_hybrid_endo"

    @classmethod
    def get_all_datasets(cls):
        dataset_names = list(DatasetRegistry)
        datasets = [dataset.value for dataset in dataset_names]
        return datasets
