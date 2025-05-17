import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# Fungsi LRU dengan OrderedDict (modifikasi print setiap 10 permintaan)
def lru_cache_simulation_with_ordereddict(requests, cache_size):
    cache = OrderedDict()
    hits = 0
    misses = 0
    for i, req in enumerate(requests):
        if req in cache:
            hits += 1
            cache.move_to_end(req)  # Pindahkan ke akhir untuk menunjukkan bahwa itu digunakan kembali
            if (i + 1) % 10 == 0:  # Cetak setiap 10 permintaan
                print(f"LRU (OrderedDict): Hit ke-{hits} (Request: {req}) - Cache: {list(cache.keys())}")
        else:
            misses += 1
            if len(cache) >= cache_size:
                cache.popitem(last=False)  # Hapus item paling lama (first in order)
            cache[req] = True  # Tambahkan item baru
            if (i + 1) % 10 == 0:  # Cetak setiap 10 permintaan
                print(f"LRU (OrderedDict): Miss ke-{misses} (Request: {req}) - Cache: {list(cache.keys())}")

    # Cek apakah requests tidak kosong sebelum menghitung hit_ratio
    hit_ratio = hits / len(requests) if len(requests) > 0 else 0
    return hit_ratio, hits, misses, list(cache.keys())

# Parameter Gaussian
mean = 50
std_dev = 10
num_requests = 1000
cache_size = 5
num_intervals = 20  # Untuk memantau perubahan hit ratio secara bertahap

# Menghasilkan data permintaan dengan distribusi Gaussian
requests = np.random.normal(mean, std_dev, num_requests).astype(int)
requests = np.clip(requests, 1, 100)  # Batasan permintaan antara 1 sampai 100

# Simulasi LRU dengan OrderedDict dan output cache serta hit/miss
print("Simulasi LRU Cache (OrderedDict):")
hit_ratio_lru, hits_lru, misses_lru, final_cache_lru = lru_cache_simulation_with_ordereddict(requests, cache_size)
print(f"Total Hits LRU: {hits_lru}, Total Misses LRU: {misses_lru}")
print(f"Cache LRU Terakhir: {final_cache_lru}\n")

# Membuat data untuk kurva Gaussian
gaussian_data = np.random.normal(mean, std_dev, num_requests)

# Membuat subplots: satu untuk kurva hit ratio, satu untuk kurva Gaussian
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Kurva garis Hit Ratio
x_values = [int(i * num_requests / num_intervals) for i in range(1, num_intervals + 1)]
hit_ratios_lru_intervals = []

# Menghitung hit ratio untuk beberapa interval permintaan
for i in range(1, num_intervals + 1):
    sample_size = int(i * num_requests / num_intervals)
    hit_ratios_lru_intervals.append(lru_cache_simulation_with_ordereddict(requests[:sample_size], cache_size)[0])

ax1.plot(x_values, hit_ratios_lru_intervals, label='LRU (OrderedDict)', color='blue', marker='o')

# Menambahkan judul, label, dan legenda
ax1.set_title('Hit Ratio LRU (OrderedDict)')
ax1.set_xlabel('Jumlah Permintaan')
ax1.set_ylabel('Hit Ratio')
ax1.legend()

# Membuat histogram untuk kurva Gaussian
count, bins, ignored = ax2.hist(gaussian_data, bins=30, density=True, alpha=0.6, color='g')

# Menghitung kurva Gaussian
gaussian_curve = (1/(std_dev * np.sqrt(2 * np.pi))) * np.exp(-(bins - mean)**2 / (2 * std_dev**2))

# Plot kurva Gaussian
ax2.plot(bins, gaussian_curve, linewidth=2, color='r')

# Memberi judul dan label
ax2.set_title('Distribusi Gaussian (Mean = 50, Std Dev = 10)')
ax2.set_xlabel('Value')
ax2.set_ylabel('Probability Density')

# Menampilkan plot
plt.tight_layout()
plt.show()
