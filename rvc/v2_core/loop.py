import json
from time import time as ttime


class EpochRecorder:
    """
    에폭 간 경과 시간을 기록
    """

    def __init__(self):
        self.last_time = ttime()

    def record(self):
        """
        경과 시간 문자열을 반환
        - 한글 주석: datetime 모듈은 호출부에서 주입
        """
        import datetime
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time = round(elapsed_time, 1)
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        return f"time={current_time} | training_speed={elapsed_time_str}"


def check_overtraining(smoothed_loss_history, threshold, epsilon=0.004):
    """
    과적합 여부 판단
    """
    if len(smoothed_loss_history) < threshold + 1:
        return False

    for i in range(-threshold, -1):
        if smoothed_loss_history[i + 1] > smoothed_loss_history[i]:
            return True
        if abs(smoothed_loss_history[i + 1] - smoothed_loss_history[i]) >= epsilon:
            return False
    return True


def update_exponential_moving_average(
    smoothed_loss_history, new_value, smoothing=0.987
):
    """
    지수이동평균(EMA) 업데이트
    """
    if smoothed_loss_history:
        smoothed_value = (
            smoothing * smoothed_loss_history[-1] + (1 - smoothing) * new_value
        )
    else:
        smoothed_value = new_value
    smoothed_loss_history.append(smoothed_value)
    return smoothed_value


def save_to_json(
    file_path,
    loss_disc_history,
    smoothed_loss_disc_history,
    loss_gen_history,
    smoothed_loss_gen_history,
):
    """
    학습 이력 JSON 저장
    """
    data = {
        "loss_disc_history": loss_disc_history,
        "smoothed_loss_disc_history": smoothed_loss_disc_history,
        "loss_gen_history": loss_gen_history,
        "smoothed_loss_gen_history": smoothed_loss_gen_history,
    }
    with open(file_path, "w") as f:
        json.dump(data, f)


