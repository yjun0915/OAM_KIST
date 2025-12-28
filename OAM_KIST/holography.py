import numpy as np
# import matplotlib.pyplot as plt


def inv_sinc(x):
    x = np.sqrt(1-x)
    y = 2*x + 3*(x**3)/10 + 321*(x**5)/2800 + 3197*(x**7)/56000 + 445617*(x**9)/13798400
    return np.sqrt(3/2) * y


def generate_oam_superposition(res, pixel_pitch, beam_w0, l_modes, weights):
    """
    OAM 중첩 상태를 위한 SLM 홀로그램 생성 (Interferogram 방식)
    """
    # 1. 좌표계 설정
    x = np.linspace(-res[0] * pixel_pitch / 2, res[0] * pixel_pitch / 2, res[0])
    y = np.linspace(-res[1] * pixel_pitch / 2, res[1] * pixel_pitch / 2, res[1])
    X, Y = np.meshgrid(x, y)

    # 극좌표 변환
    R = np.sqrt(X ** 2 + Y ** 2)
    Phi = np.arctan2(Y, X)

    # 2. 빔 프로파일 정의 (Laguerre-Gaussian 근사)
    # 0으로 나누는 것 방지
    R[R == 0] = 1e-10

    # 3. 중첩 (Superposition) - 여기서 복소수 합이 일어납니다.
    E_total = np.zeros_like(Phi, dtype=complex)
    for l, w in zip(l_modes, weights):
        # 각 모드의 복소수 필드 생성 (LG mode, p=0 가정)
        # E = (sqrt(2)r/w)^|l| * exp(-r^2/w^2) * exp(il*phi)
        E_total += w * (np.sqrt(2) * R / beam_w0) ** abs(l) * np.exp(-R ** 2 / beam_w0 ** 2) * np.exp(1j * l * Phi)


    # 목표 진폭(Amplitude)과 위상(Phase) 추출
    Amp = np.abs(E_total)
    Phase = np.angle(E_total)

    # 진폭 정규화 (0 ~ 1)
    Amp = Amp / np.max(Amp)

    return Amp, Phase, X, Y


def encode_hologram(Amp, Phase, X, Y, grating_d, grating_N, pixel_pitch):
    """
    진폭 정보를 포함하여 SLM에 띄울 최종 Phase Mask 생성
    방식: Off-axis Holography (Carrier frequency 추가)
    """

    modified_Amp = 1 + (1/np.pi) * inv_sinc(Amp)

    modified_Amp = modified_Amp / np.max(modified_Amp)
    modified_Phase = Phase - np.pi*modified_Amp

    # Blazed Grating (Carrier Frequency) 생성
    # k_tilt * x
    carrier_phase = (2 * np.pi * X) / (2*np.pi*grating_d)

    phase_depth = 2 * np.arcsin(Amp)  # 진폭을 위상 깊이로 변환
    hologram_pattern = phase_depth * np.cos(Phase + carrier_phase)

    # SLM 0~2pi 매핑 (Gray scale)
    hologram_final = (np.mod(hologram_pattern, 2*np.pi)/(2*np.pi/grating_N)).astype(int)/(grating_N-1)

    return phase_depth

