import multem
import pickle


def test_system_configuration():
    system_conf = multem.SystemConfiguration()
    system_conf.device = "device"
    system_conf.precision = "precision"
    system_conf.cpu_ncores = 1
    system_conf.cpu_nthread = 1
    system_conf.gpu_device = 1
    system_conf.gpu_nstream = 1

    def check():
        assert system_conf.device == "device"
        assert system_conf.precision == "precision"
        assert system_conf.cpu_ncores == 1
        assert system_conf.cpu_nthread == 1
        assert system_conf.gpu_device == 1
        assert system_conf.gpu_nstream == 1

    check()

    system_conf = pickle.loads(pickle.dumps(system_conf))

    check()
