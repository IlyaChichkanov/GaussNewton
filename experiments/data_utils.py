import numpy as np
from jax import jit
from jax import numpy as jnp
from pathlib import Path
import pandas as pd
from typing import Callable
from scipy.interpolate import interp1d

def create_interval_batches(time,  f : Callable, max_delta_t = 0.5, min_interval = 4.0):
    state_batches = []
    t_batches = []

    state_batch = []
    time_batch = []

    t_prev = time[0]
    min_seq_length = 2
    for i, t in enumerate(time):
        batch_ready = len(time_batch) >= min_seq_length and abs(time_batch[-1] - time_batch[0]) > min_interval

        if(np.abs(t - t_prev) < max_delta_t) :
            state_batch.append(f(t))
            time_batch.append(t)
        else:
            if(batch_ready):
                data_batch = np.copy(state_batch)#[:, np.newaxis]
                print(data_batch.shape)
                if(len(data_batch.shape) > 2):
                    data_batch = data_batch.squeeze()
                state_batches.append(data_batch)
                t_batches.append(np.copy(np.array(time_batch)))
            time_batch = []
            state_batch = []
        t_prev = t

        if(batch_ready and i == len(time) - 1):
            data_batch = np.copy(state_batch)#[:, np.newaxis]
            if(len(data_batch.shape) > 2):
                data_batch = data_batch.squeeze()
            state_batches.append(data_batch)
            t_batches.append(np.copy(np.array(time_batch)))

    return t_batches, state_batches

def create_jax_interpolator(x_data, y_data, method='linear'):
    """
    Create an interpolator that works with odeint by using only JAX operations
    """
    x_jax = jnp.array(x_data)
    y_jax = jnp.array(y_data)
    
    if method == 'linear':
        @jit
        def interpolator(x, *p):
            # Find indices for interpolation
            idx = jnp.searchsorted(x_jax, x)
            idx = jnp.clip(idx, 1, len(x_jax) - 1)
            
            # Get surrounding points
            x0 = x_jax[idx - 1]
            x1 = x_jax[idx]
            y0 = y_jax[idx - 1]
            y1 = y_jax[idx]
            
            # Linear interpolation
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        
        return interpolator
    
    elif method == 'nearest':
        @jit
        def interpolator(x):
            # Vectorized nearest neighbor
            diff = jnp.abs(x_jax - x[:, None])
            indices = jnp.argmin(diff, axis=1)
            return y_jax[indices]
        
        return interpolator


def theta_to_physical(theta, m, L, g=9.81):
    """Нормированные параметры велосипедной модели -> физические величины.

    theta: [Cf_norm, Cr_norm, a_rel, Iz_norm] и, опционально, steering_ratio
    пятым элементом (тогда он попадает в результат как 'steering_ratio').
    m — масса, L — колёсная база.
    """
    Cf_norm, Cr_norm, a_rel, Iz_norm = theta[:4]
    a = a_rel * L
    physical = {
        'Cf': Cf_norm * m * g,
        'Cr': Cr_norm * m * g,
        'a': a,
        'b': L - a,
        'Iz': Iz_norm * m * L * L,
    }
    if len(theta) > 4:
        physical['steering_ratio'] = theta[4]
    return physical


from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Предполагается, что где-то определена функция create_jax_interpolator
# from your_module import create_jax_interpolator

class LogReaderV2:
    def __init__(self, paths):
        """
        paths: Path, список путей или словарь {имя: путь}
        """
        if isinstance(paths, (str, Path)):
            self.source_paths = {'default': Path(paths)}
        elif isinstance(paths, list):
            self.source_paths = {f'source_{i}': Path(p) for i, p in enumerate(paths)}
        elif isinstance(paths, dict):
            self.source_paths = {k: Path(v) for k, v in paths.items()}
        else:
            raise TypeError("paths должен быть Path, список или словарь")

        self.dfs = {}
        self.data = {}
        self._pending_batches = []

        self.t0 = 0
        self.t1 = -np.inf
        self.t2 = np.inf

        for src_name, p in self.source_paths.items():
            if not p.exists():
                raise ValueError(f"Файл не найден: {p}")
            df = pd.read_csv(p)
            if df.empty:
                raise ValueError(f"Нет данных в файле: {p}")
            if 'time' not in df.columns:
                raise ValueError(f"В файле {p} нет колонки 'time'")
            self.dfs[src_name] = df
            print(f"Загружен {src_name}: {len(df)} строк из {p}")

    def add_batch(self, dataname: str, source: str = None,
                  time_ns: bool = False, use_jax_interp=True, scale_coef=1):
        if source is None:
            source = next(iter(self.dfs.keys()))
        if source not in self.dfs:
            raise ValueError(f"Источник '{source}' не найден. Доступные: {list(self.dfs.keys())}")
        df = self.dfs[source]
        if dataname not in df.columns:
            raise ValueError(f"Колонка '{dataname}' не найдена в источнике '{source}'.\n"
                             f"Доступные колонки: {list(df.columns)}")

        time = df["time"].values.copy().astype(float)
        values = df[dataname].values.copy().astype(float) * scale_coef
        valid = ~(np.isnan(time) | np.isnan(values))
        time = time[valid]
        values = values[valid]
        if len(time) == 0:
            raise ValueError(f"Нет корректных данных для {dataname} в {source}")
        if time_ns:
            time = time / 1e9

        self._pending_batches.append({
            'name': dataname,
            'source': source,
            'time_original': time,
            'values': values,
            'use_jax_interp': use_jax_interp,
            'time_ns': time_ns,
            'scale_coef': scale_coef
        })
        print(f"Добавлен в очередь {dataname} из {source}: "
              f"time [{time[0]:.3f}, {time[-1]:.3f}], "
              f"values [{values.min():.3f}, {values.max():.3f}]")

    def process_all(self):
        if not self._pending_batches:
            print("Нет данных для обработки")
            return

        all_starts = [b['time_original'][0] for b in self._pending_batches]
        self.t0 = min(all_starts)
        print(f"\nОбщий t0 = {self.t0}")

        for batch in self._pending_batches:
            tn = batch['time_original'] - self.t0
            if batch['use_jax_interp']:
                f = create_jax_interpolator(tn, batch['values'], method='linear')
                print("  используем jax")
            else:
                f = interp1d(tn, batch['values'], kind='linear',
                             bounds_error=False, fill_value='extrapolate')

            self.data[batch['name']] = {
                'interpolator': f,
                'time_original': batch['time_original'],
                'time': tn,
                'values': batch['values'],
                'source': batch['source'],
                'time_ns': batch['time_ns'],
                'scale_coef': batch['scale_coef']
            }
            self.t1 = max(self.t1, tn[0])
            self.t2 = min(self.t2, tn[-1])
            print(f"Обработан {batch['name']}: норм. время [{tn[0]:.3f}, {tn[-1]:.3f}]")

        self._pending_batches.clear()
        print(f"Общий временной интервал: [{self.t1:.3f}, {self.t2:.3f}]")

    # ========== ГЕТТЕРЫ (без изменений) ==========
    def get_f_interp(self, dataname: str):
        if dataname not in self.data:
            raise ValueError(f"Данные '{dataname}' не загружены. Доступные: {list(self.data.keys())}")
        return self.data[dataname]['interpolator']

    def get_time(self, dataname: str):
        if dataname not in self.data:
            raise ValueError(f"Данные '{dataname}' не загружены")
        return self.data[dataname]['time']

    def get_original_time(self, dataname: str):
        if dataname not in self.data:
            raise ValueError(f"Данные '{dataname}' не загружены")
        return self.data[dataname]['time_original']

    def get_raw_data(self, dataname: str):
        if dataname not in self.data:
            raise ValueError(f"Данные '{dataname}' не загружены")
        return self.data[dataname]['values'], self.data[dataname]['time']

    def get_data_at_time(self, dataname: str, t):
        f = self.get_f_interp(dataname)
        return float(f(t))

    def get_common_time_range(self):
        return self.t1, self.t2

    def get_available_data(self):
        return list(self.data.keys())

    def plot_data(self):
        if not self.data:
            print("Нет данных для визуализации")
            return

        fig, axes = plt.subplots(len(self.data), 1, figsize=(12, 3*len(self.data)))
        if len(self.data) == 1:
            axes = [axes]

        for ax, (name, d) in zip(axes, self.data.items()):
            ax.plot(d['time'], d['values'], label=name)
            ax.set_xlabel('Time (normalized)')
            ax.set_ylabel(name)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

