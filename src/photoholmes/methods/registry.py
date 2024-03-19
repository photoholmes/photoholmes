from enum import Enum, unique


@unique
class MethodRegistry(Enum):
    NAIVE = "naive"
    DQ = "dq"
    SPLICEBUSTER = "splicebuster"
    CATNET = "catnet"
    EXIF_AS_LANGUAGE = "exif_as_language"
    ADAPTIVE_CFA_NET = "adaptive_cfa_net"
    NOISESNIFFER = "noisesniffer"
    PSCCNET = "psccnet"
    TRUFOR = "trufor"
    FOCAL = "focal"
    ZERO = "zero"

    @classmethod
    def get_all_methods(cls):
        methods_names = list(MethodRegistry)
        methods = [method.value for method in methods_names]
        return methods
