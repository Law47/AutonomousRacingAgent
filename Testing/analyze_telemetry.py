import pandas as pd
import numpy as np

df = pd.read_csv(r'Trainer\TelemetryData\ACSharedMemory_20260204_205205.csv')
print(f'Total rows: {len(df)}')
print(f'First packet ID: {df["packetId"].iloc[0]}')
print(f'Last packet ID: {df["packetId"].iloc[-1]}')
print(f'Packet ID range: {df["packetId"].iloc[-1] - df["packetId"].iloc[0] + 1}')

# Calculate gaps
packet_ids = df['packetId'].values
gaps = np.diff(packet_ids)
gap_counts = pd.Series(gaps).value_counts().sort_index()
print(f'\nPacket ID gaps:')
print(gap_counts)

print(f'\nGap analysis:')
print(f'Expected gap (consecutive): 1')
print(f'Number of gaps == 1: {(gaps == 1).sum()}')
print(f'Number of gaps > 1: {(gaps > 1).sum()}')
print(f'Total frames lost: {gaps.sum() - len(gaps)}')

# Time analysis
df['timestamp'] = pd.to_datetime(df['timestamp'])
time_diffs = df['timestamp'].diff().dt.total_seconds().dropna()
print(f'\nTime between frames (ms):')
print(f'Mean: {time_diffs.mean() * 1000:.4f}')
print(f'Std: {time_diffs.std() * 1000:.4f}')
print(f'Min: {time_diffs.min() * 1000:.4f}')
print(f'Max: {time_diffs.max() * 1000:.4f}')

# Count frames per second
df['second'] = df['timestamp'].dt.floor('S')
fps = df.groupby('second').size()
print(f'\nFrames per second:')
print(f'Mean: {fps.mean():.1f}')
print(f'Min: {fps.min()}')
print(f'Max: {fps.max()}')
print(f'Distribution:')
print(fps.value_counts().sort_index().tail(10))

# Calculate effective recording rate
total_time = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
effective_hz = len(df) / total_time if total_time > 0 else 0
print(f'\nEffective recording rate:')
print(f'Duration: {total_time:.2f} seconds')
print(f'Effective Hz: {effective_hz:.2f}')
print(f'Expected at 333 Hz for {len(df)} frames: {len(df) / 333:.2f} seconds')
