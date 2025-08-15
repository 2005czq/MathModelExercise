import math

# 中心点坐标
center_lat = 25.0
center_lon = 120.0

# 范围边长 (公里)
side_length_km = 20.0
half_side_km = side_length_km / 2.0

# --- 计算 ---

# 1. 计算纬度范围
# 1度纬度的距离近似为 111.1 公里
km_per_degree_lat = 111.1
lat_delta = half_side_km / km_per_degree_lat

lat_north = center_lat + lat_delta
lat_south = center_lat - lat_delta

# 2. 计算经度范围
# 1度经度的距离取决于纬度
km_per_degree_lon = 111.320 * math.cos(math.radians(center_lat))
lon_delta = half_side_km / km_per_degree_lon

lon_east = center_lon + lon_delta
lon_west = center_lon - lon_delta

# --- 输出结果 ---
print(f"中心点: ({center_lat}°N, {center_lon}°E)")
print(f"范围: {side_length_km}km * {side_length_km}km")
print("-" * 30)
print(f"纬度范围 (Latitude):")
print(f"  北界 (North): {lat_north:.4f}°N")
print(f"  南界 (South): {lat_south:.4f}°N")
print(f"经度范围 (Longitude):")
print(f"  东界 (East): {lon_east:.4f}°E")
print(f"  西界 (West): {lon_west:.4f}°E")
