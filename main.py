from OAM_KIST.holography import generate_oam_superposition, encode_hologram
import cv2



res = [1920, 1080]  # SLM 해상도 (예: 1080p)
pitch = 8e-6  # 픽셀 피치 (8um)
w0 = 0.0008  # 빔 웨이스트 (0.8mm)
l_modes = [-3, -1, 1, 8]  # 첫 번째 모드
weights = [1, 0, 0, 0]  # 두 번째 모드 (부호 반대)
grating_period = 80e-6  # 회절 격자 주기 (분리 각도 결정)
d, N = 64*pitch, 8

# --- 실행 ---
amp_map, phase_map, X, Y = generate_oam_superposition(res, pitch, w0, l_modes, weights)
slm_hologram = encode_hologram(amp_map, phase_map, X, Y, d, N, pitch)

print(slm_hologram.max())
cv2.imshow("hologram", slm_hologram)
cv2.waitKey(0)