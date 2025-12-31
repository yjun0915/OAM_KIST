import pytest
import numpy as np
import os
from OAM_KIST.holography import inv_sinc, generate_oam_superposition, encode_hologram


# 1. 수식 정확성 테스트 (Docstring 예제 검증)
# def test_inv_sinc_accuracy():
#     """inv_sinc 함수가 예상된 수학적 값을 반환하는지 테스트"""
#     x_val = 0.5 * np.pi
#     expected = 1.8947036302816116
#     result = inv_sinc(x_val)
#
#     # 부동소수점 계산이므로 == 대신 np.isclose 사용 (오차 허용)
#     assert np.isclose(result, expected, atol=1e-6)


# 2. OAM 생성 함수 입출력 형태 테스트
def test_generate_oam_superposition_shape():
    """OAM 중첩 함수가 올바른 크기의 배열을 반환하는지 테스트"""
    res = [100, 100]  # 테스트니까 작게 설정
    pixel_pitch = 8e-6
    beam_w0 = 8e-4
    l_modes = [-1, 1]
    weights = [0.5, 0.5]

    amp, phase, X, Y = generate_oam_superposition(res, pixel_pitch, beam_w0, l_modes, weights)

    # 반환된 배열의 모양(Shape)이 해상도와 일치하는지 확인
    assert amp.shape == (100, 100)
    assert phase.shape == (100, 100)
    assert X.shape == (100, 100)
    assert Y.shape == (100, 100)

    # 값의 범위 확인 (물리적 의미 검증)
    assert np.max(amp) <= 1.0 + 1e-9  # 부동소수점 오차 고려
    assert np.min(amp) >= 0.0


# 3. 홀로그램 생성 및 파일 저장 테스트
def test_encode_hologram_save(tmp_path):
    """
    tmp_path: pytest가 제공하는 임시 폴더 Fixture (테스트 끝나면 자동 삭제됨)
    실제로 파일을 저장하고 잘 생기는지 테스트합니다.
    """
    # 더미 데이터 생성
    res = 50
    X, Y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
    amp = np.random.rand(res, res)
    phase = np.random.rand(res, res) * 2 * np.pi - np.pi

    save_dir = tmp_path / "test_outputs"
    file_name = "test_hologram"

    # 함수 실행 (save=True)
    encode_hologram(
        Amp=amp, Phase=phase, X=X, Y=Y,
        pixel_pitch=1e-6, d=10,
        prepare=True, save=True,
        path=str(save_dir), name=file_name
    )

    # 파일이 진짜 생성되었는지 확인
    expected_file = save_dir / (file_name + ".png")
    assert expected_file.exists()