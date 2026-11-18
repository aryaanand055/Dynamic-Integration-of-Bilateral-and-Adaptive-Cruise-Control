"""Run the committed TD3 actor (SB3, net_arch [400,300], tanh, action range +-5) with numpy only,
so the scenario runs need neither PyTorch nor stable-baselines3. Also installs a tiny
stable_baselines3 stub so repo modules that import it can load."""
import sys, types, zipfile, pickle, io, json
import numpy as np

def _stub_sb3():
    if 'stable_baselines3' in sys.modules:
        return
    sb3 = types.ModuleType('stable_baselines3'); sb3.TD3 = object
    common = types.ModuleType('stable_baselines3.common'); ve = types.ModuleType('stable_baselines3.common.vec_env')
    ve.VecEnv = object
    sys.modules.update({'stable_baselines3': sb3, 'stable_baselines3.common': common, 'stable_baselines3.common.vec_env': ve})
_stub_sb3()

def _load_state(pth_bytes):
    zf = zipfile.ZipFile(io.BytesIO(pth_bytes)); root = zf.namelist()[0].split('/')[0]
    dtypes = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'LongStorage': np.int64}
    def rebuild(storage, offset, size, stride, *a):
        arr = storage
        n = int(np.prod(size)) if size else 1
        flat = arr[offset:offset + n] if not stride or list(stride) == list(np.cumprod((list(size)[1:] + [1])[::-1])[::-1]) else None
        return (flat if flat is not None else arr[offset:offset + n]).reshape(size)
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == '_rebuild_tensor_v2': return rebuild
            if mod == 'collections' and name == 'OrderedDict':
                import collections; return collections.OrderedDict
            if mod.startswith('torch') and name.endswith('Storage'): return name
            return lambda *a, **k: None
        def persistent_load(self, pid):
            _, stype, key, loc, numel = pid
            raw = zf.read(f'{root}/data/{key}')
            return np.frombuffer(raw, dtype=dtypes.get(stype, np.float32)).copy()
    return U(io.BytesIO(zf.read(f'{root}/data.pkl'))).load()

class NumpyTD3:
    def __init__(self, zip_path):
        z = zipfile.ZipFile(zip_path)
        sd = _load_state(z.read('policy.pth'))
        self.W = [(sd[f'actor.mu.{i}.weight'], sd[f'actor.mu.{i}.bias']) for i in (0, 2, 4)]
        d = json.loads(z.read('data'))
        self.low, self.high = -5.0, 5.0
        class _S: pass
        self.observation_space = _S(); self.observation_space.shape = (self.W[0][0].shape[1],)
    def predict(self, obs, deterministic=True):
        x = np.asarray(obs, dtype=np.float32)
        for k, (w, b) in enumerate(self.W):
            x = x @ w.T + b
            x = np.maximum(x, 0) if k < 2 else np.tanh(x)
        act = self.low + 0.5 * (x + 1.0) * (self.high - self.low)
        return act, None
