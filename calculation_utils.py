"""
생산 데이터 계산 유틸리티 모듈

이 모듈은 생산 데이터 분석에 사용되는 모든 핵심 계산 로직을 제공합니다.
단일 진실 공급원(Single Source of Truth)으로 작동하며,
analyzer_v4.1.py와 DashBoard_V46_cursor_V024.py 모두에서 사용됩니다.

주요 계산 로직:
1. 수율 계산 (단순 수율, 종합 수율, 로트별 종합수율)
2. 가동률 계산
3. 목표 달성률 계산
4. 불량률 계산
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np


# =============================================================================
# 1. 수율 계산 (Yield Calculation)
# =============================================================================

def calculate_simple_yield(good_qty: float, total_qty: float, as_percentage: bool = True) -> float:
    """
    단순 수율 계산: 양품수량 / 생산수량

    Args:
        good_qty: 양품수량
        total_qty: 총 생산수량
        as_percentage: True면 백분율(0-100), False면 소수점(0-1)

    Returns:
        수율 값 (백분율 또는 소수점)

    Examples:
        >>> calculate_simple_yield(95, 100)  # 95%
        95.0
        >>> calculate_simple_yield(95, 100, as_percentage=False)  # 0.95
        0.95

    특수 케이스:
        - 생산수량이 0인 경우: 0 반환 (ZeroDivisionError 방지)
        - 수율이 100%를 초과하는 경우: 그대로 반환 (데이터 이상 표시 목적)
    """
    if total_qty == 0:
        return 0.0

    yield_value = good_qty / total_qty

    if as_percentage:
        return round(yield_value * 100, 2)
    else:
        return round(yield_value, 4)


def calculate_overall_yield_by_multiplication(process_yields: pd.Series) -> float:
    """
    종합 수율 계산 (곱셈 방식): 각 공정 수율의 곱

    제조업에서 제품이 여러 공정을 거칠 때, 최종 수율은 각 공정 수율의 곱으로 계산됩니다.
    예: 공정1(95%) × 공정2(98%) × 공정3(97%) = 전체 수율(90.3%)

    Args:
        process_yields: 각 공정의 수율 (0-1 사이 값, 예: 0.95 = 95%)

    Returns:
        종합 수율 (백분율, 0-100)

    Examples:
        >>> yields = pd.Series([0.95, 0.98, 0.97])
        >>> calculate_overall_yield_by_multiplication(yields)
        90.3

    특수 케이스:
        - 공정 수율이 비어있는 경우: 0 반환
        - 하나의 공정이라도 0%인 경우: 전체가 0%
        - 수율이 100% 초과인 경우: clip 처리 없이 그대로 계산 (데이터 검증 목적)
    """
    if process_yields.empty:
        return 0.0

    overall = process_yields.prod()
    return round(overall * 100, 2)


def calculate_lot_based_overall_yield(
    df: pd.DataFrame,
    lot_id_col: str = 'CHECK SHEET NO',
    process_col: str = '공정코드',
    production_col: str = '생산수량',
    good_col: str = '양품수량'
) -> pd.DataFrame:
    """
    로트별 종합 수율 계산 (가중 평균 방식)

    이 함수는 analyzer의 핵심 로직을 구현합니다:
    1. 로트별, 공정별 수율 계산
    2. 각 로트의 종합 수율 계산 (공정 수율의 곱)
    3. 생산수량 기반 가중 평균으로 최종 종합 수율 계산

    Args:
        df: 생산 데이터 DataFrame
        lot_id_col: 로트 ID 컬럼명 (기본: 'CHECK SHEET NO')
        process_col: 공정코드 컬럼명
        production_col: 생산수량 컬럼명
        good_col: 양품수량 컬럼명

    Returns:
        원본 DataFrame에 '로트별 종합수율' 컬럼이 추가된 DataFrame

    로직 상세:
        1. 로트별, 공정별 집계: 생산수량과 양품수량을 합산
        2. 공정별 수율 계산: 양품수량 / 생산수량
        3. 수율 보정: 100% 초과 시 100%로 clip (데이터 품질 보정)
        4. 로트별 종합 수율: 각 공정 수율의 곱 (제조 공정 특성)
        5. 원본 데이터에 병합: left join으로 모든 행에 종합 수율 추가

    특수 케이스:
        - CHECK SHEET NO가 없는 경우: 원본 DataFrame 그대로 반환
        - CHECK SHEET NO가 모두 NA인 경우: 원본 DataFrame 그대로 반환
        - 생산수량이 0인 경우: 해당 공정 수율을 0으로 설정
        - 수율이 100% 초과: 1.0(100%)으로 clip 처리

    Examples:
        데이터 예시:
        CHECK SHEET NO  공정코드  생산수량  양품수량
        LOT001         [10]    1000     950
        LOT001         [20]     950     930
        LOT001         [80]     930     910

        계산:
        - [10] 수율: 950/1000 = 0.95
        - [20] 수율: 930/950 = 0.979
        - [80] 수율: 910/930 = 0.978
        - 종합 수율: 0.95 × 0.979 × 0.978 = 0.909 (90.9%)
    """
    # CHECK SHEET NO 컬럼 확인
    if lot_id_col not in df.columns or df[lot_id_col].isna().all():
        return df

    # 1. 로트별, 공정별 수율 계산
    lot_process_yields = df.groupby([lot_id_col, process_col]).agg(
        생산수량_lot=(production_col, 'sum'),
        양품수량_lot=(good_col, 'sum')
    ).reset_index()

    # 2. 공정별 수율 계산 (0으로 나누기 방지)
    lot_process_yields['수율'] = 0.0
    mask = lot_process_yields['생산수량_lot'] != 0
    lot_process_yields.loc[mask, '수율'] = (
        lot_process_yields.loc[mask, '양품수량_lot'] /
        lot_process_yields.loc[mask, '생산수량_lot']
    )

    # 3. 수율이 100%를 초과하는 경우 100%로 보정
    lot_process_yields['수율'] = lot_process_yields['수율'].clip(upper=1.0)

    # 4. 로트별 종합 수율 계산 (각 공정 수율의 곱)
    lot_overall_yield = lot_process_yields.groupby(lot_id_col)['수율'].prod().reset_index()
    lot_overall_yield.rename(columns={'수율': '로트별 종합수율'}, inplace=True)

    # 5. 원본 데이터에 로트별 종합 수율 병합
    result_df = pd.merge(df, lot_overall_yield, on=lot_id_col, how='left')

    return result_df


def calculate_weighted_overall_yield(
    df: pd.DataFrame,
    group_cols: list,
    lot_yield_col: str = '로트별 종합수율',
    production_col: str = '생산수량'
) -> pd.DataFrame:
    """
    생산수량 기반 가중 평균 종합 수율 계산

    여러 로트를 집계할 때, 단순 평균이 아닌 생산수량 기반 가중 평균을 사용합니다.
    생산량이 많은 로트의 수율이 더 큰 영향을 미치도록 합니다.

    Args:
        df: 로트별 종합수율이 포함된 DataFrame
        group_cols: 집계 기준 컬럼 리스트 (예: ['공장', '공정코드'])
        lot_yield_col: 로트별 종합수율 컬럼명
        production_col: 생산수량 컬럼명

    Returns:
        집계된 DataFrame (group_cols + '로트별 종합수율')

    계산 공식:
        가중 평균 수율 = Σ(로트별 종합수율 × 생산수량) / Σ(생산수량)

    Examples:
        LOT001: 수율 90%, 생산 1000개
        LOT002: 수율 95%, 생산 500개
        가중 평균 = (0.9×1000 + 0.95×500) / (1000+500) = 91.67%

        단순 평균 = (90 + 95) / 2 = 92.5% (부정확!)

    특수 케이스:
        - 전체 생산수량이 0인 경우: 0 반환
        - 로트별 종합수율이 없는 경우: 빈 DataFrame 반환
    """
    def weighted_avg(group):
        total_production = group[production_col].sum()
        if total_production == 0:
            return 0
        weighted_sum = (group[lot_yield_col] * group[production_col]).sum()
        return weighted_sum / total_production

    result = df.groupby(group_cols).apply(weighted_avg).reset_index(name=lot_yield_col)
    return result


# =============================================================================
# 2. 가동률 계산 (Utilization Rate Calculation)
# =============================================================================

def calculate_utilization_rate(
    production_qty: float,
    max_capacity: float,
    as_percentage: bool = True
) -> float:
    """
    가동률 계산: 실제 생산수량 / 이론상 최대 생산량

    Args:
        production_qty: 실제 생산수량
        max_capacity: 이론상 최대 생산량
        as_percentage: True면 백분율(0-100), False면 소수점(0-1)

    Returns:
        가동률 값

    Examples:
        >>> calculate_utilization_rate(850, 1000)
        85.0

    특수 케이스:
        - 최대 생산량이 0인 경우: 0 반환 (설비 정보 없음)
        - 가동률이 100% 초과: 그대로 반환 (초과 근무 등 실제 상황)
    """
    if max_capacity == 0:
        return 0.0

    utilization = production_qty / max_capacity

    if as_percentage:
        return round(utilization * 100, 2)
    else:
        return round(utilization, 4)


def calculate_theoretical_max_production(
    daily_max: float,
    operating_days: int
) -> float:
    """
    이론상 총 생산량 계산

    Args:
        daily_max: 일일 최대 생산량
        operating_days: 운영 일수

    Returns:
        이론상 총 생산량

    Examples:
        >>> calculate_theoretical_max_production(1000, 7)  # 주간 최대
        7000
    """
    return daily_max * operating_days


# =============================================================================
# 3. 목표 달성률 계산 (Target Achievement Rate Calculation)
# =============================================================================

def calculate_target_achievement_rate(
    actual_qty: float,
    target_qty: float,
    basis: str = 'good',
    as_percentage: bool = True
) -> float:
    """
    목표 달성률 계산

    Args:
        actual_qty: 실제 수량 (양품수량 또는 생산수량)
        target_qty: 목표 수량
        basis: 'good'(양품 기준) 또는 'production'(생산 기준)
        as_percentage: True면 백분율

    Returns:
        달성률

    중요:
        - **양품 기준 달성률**을 주로 사용 (품질 포함 성과)
        - 생산수량 기준은 참고용

    Examples:
        >>> calculate_target_achievement_rate(950, 1000, basis='good')
        95.0

    특수 케이스:
        - 목표가 0인 경우: 0 반환
        - 달성률 100% 초과: 그대로 반환 (초과 달성 표시)
    """
    if target_qty == 0:
        return 0.0

    achievement = actual_qty / target_qty

    if as_percentage:
        return round(achievement * 100, 2)
    else:
        return round(achievement, 4)


def calculate_total_target(
    daily_target: float,
    operating_days: int
) -> float:
    """
    목표 총 생산량 계산

    Args:
        daily_target: 일일 생산목표량
        operating_days: 운영 일수

    Returns:
        목표 총 생산량
    """
    return daily_target * operating_days


# =============================================================================
# 4. 불량률 계산 (Defect Rate Calculation)
# =============================================================================

def calculate_defect_rate(
    defect_qty: float,
    total_qty: float,
    as_percentage: bool = True
) -> float:
    """
    불량률 계산: 불량수량 / 생산수량

    Args:
        defect_qty: 불량수량
        total_qty: 총 생산수량
        as_percentage: True면 백분율

    Returns:
        불량률

    관계:
        수율 + 불량률 = 100% (양품수량 + 불량수량 = 생산수량)

    Examples:
        >>> calculate_defect_rate(5, 100)
        5.0
    """
    if total_qty == 0:
        return 0.0

    defect_rate = defect_qty / total_qty

    if as_percentage:
        return round(defect_rate * 100, 2)
    else:
        return round(defect_rate, 4)


# =============================================================================
# 5. 판다스 DataFrame 전용 헬퍼 함수
# =============================================================================

def safe_divide_series(
    numerator: pd.Series,
    denominator: pd.Series,
    fill_value: float = 0.0
) -> pd.Series:
    """
    안전한 나눗셈 (0으로 나누기 방지)

    Args:
        numerator: 분자 Series
        denominator: 분모 Series
        fill_value: 0으로 나눴을 때 채울 값

    Returns:
        나눗셈 결과 Series

    Examples:
        >>> numerator = pd.Series([100, 200, 300])
        >>> denominator = pd.Series([10, 0, 30])
        >>> safe_divide_series(numerator, denominator)
        0    10.0
        1     0.0  # 0으로 나눈 경우
        2    10.0
    """
    # where를 사용하여 분모가 0이 아닐 때만 나눗셈 수행
    result = numerator / denominator.where(denominator != 0)
    return result.fillna(fill_value)


def add_yield_column(
    df: pd.DataFrame,
    production_col: str = '총_생산수량',
    good_col: str = '총_양품수량',
    result_col: str = '전체_수율(%)'
) -> pd.DataFrame:
    """
    DataFrame에 수율 컬럼 추가

    Args:
        df: DataFrame
        production_col: 생산수량 컬럼명
        good_col: 양품수량 컬럼명
        result_col: 결과 컬럼명

    Returns:
        수율 컬럼이 추가된 DataFrame
    """
    df[result_col] = round(
        safe_divide_series(df[good_col], df[production_col]) * 100, 2
    )
    return df


def add_utilization_column(
    df: pd.DataFrame,
    production_col: str = '총_생산수량',
    capacity_col: str = '이론상_총_생산량',
    result_col: str = '가동률(%)'
) -> pd.DataFrame:
    """
    DataFrame에 가동률 컬럼 추가
    """
    df[result_col] = round(
        safe_divide_series(df[production_col], df[capacity_col]) * 100, 2
    )
    return df


def add_achievement_column(
    df: pd.DataFrame,
    actual_col: str = '총_양품수량',
    target_col: str = '목표_총_생산량',
    result_col: str = '양품수_기준_달성률(%)'
) -> pd.DataFrame:
    """
    DataFrame에 목표 달성률 컬럼 추가
    """
    df[result_col] = round(
        safe_divide_series(df[actual_col], df[target_col]) * 100, 2
    )
    return df


# =============================================================================
# 6. 검증 및 보정 함수
# =============================================================================

def clip_percentage(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    """
    백분율 값을 특정 범위로 제한

    Args:
        value: 원본 값
        lower: 최소값
        upper: 최대값

    Returns:
        제한된 값

    Note:
        일반적으로는 제한하지 않지만, 데이터 품질 보정이 필요한 경우 사용
    """
    return max(lower, min(upper, value))


def validate_production_data(
    df: pd.DataFrame,
    required_cols: list
) -> Tuple[bool, str]:
    """
    생산 데이터 유효성 검증

    Args:
        df: 검증할 DataFrame
        required_cols: 필수 컬럼 리스트

    Returns:
        (검증 성공 여부, 오류 메시지)

    Examples:
        >>> is_valid, msg = validate_production_data(df, ['생산수량', '양품수량'])
        >>> if not is_valid:
        >>>     print(msg)
    """
    if df.empty:
        return False, "데이터가 비어있습니다."

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"필수 컬럼이 없습니다: {', '.join(missing_cols)}"

    return True, ""
