from typing import Union
import re
import torch
import numpy as np
from utils import load_json, load_csv, text_to_time, get_logger, pickle_writer, csv_writer

transform_logger = get_logger('transform')
phase_balances = {"train": 0, "valid": 0, "test": 0}
phase_counts = {"train": 0, "valid": 0, "test": 0}
phase_ratio = {"train": 8, "valid": 1, "test": 1}
prev_genre = None
experiment_result = None
playlist_path = "../configs/tracklist.csv"


def _detect_phase(genre: str) -> str:
    global prev_genre, phase_balances, phase_counts

    if prev_genre is not None and genre != prev_genre:
        phase_balances = {"train": 0, "valid": 0, "test": 0}

    current_ratio = {"train": phase_balances["train"] / phase_ratio["train"], "valid": phase_balances["valid"] /
                     phase_ratio["valid"], "test": phase_balances["test"] / phase_ratio["test"]}
    min_key = min(current_ratio, key=lambda k: current_ratio[k])
    phase_balances[min_key] += 1
    phase_counts[min_key] += 1
    prev_genre = genre

    return min_key


def _get_task_id(task: str) -> Union[int, None]:
    if task == 'Vocal':
        return 0
    elif task == 'Drums':
        return 1
    elif task == 'Bass':
        return 2
    elif task == 'Other(Drums, Bass, Vocal以外の楽器)':
        return 3
    else:
        return None


def _convert_single(text: str) -> Union[float, None]:
    try:
        text = text.strip()
        text = text.replace(',', '')
        if text.endswith('%'):
            return float(text.rstrip('%')) / 100

        return float(text)

    except ValueError as e:
        transform_logger.warning(f"Failed to convert: {text}, error: {str(e)}")
        return None


def _is_noisy_data(eeg_list: list) -> bool:
    eeg_arr = np.array(eeg_list)
    return np.any(eeg_arr > 2.0)


def _get_experiment_data(experiment: str) -> dict:
    global experiment_result

    if experiment_result is None:
        experiment_result = load_json(experiment)
        experiment_result = sorted(experiment_result, key=lambda x: x['Time'])

    return experiment_result


def _get_survey_result(experiment: str) -> dict:
    experiment_result = _get_experiment_data(experiment)
    survey_result_list = {x["Event"].replace(
        "survey", ""): x for x in experiment_result if re.search(r"survey[0-9]", x["Event"])}
    return survey_result_list


def _get_experiment_result(experiment: str) -> dict:
    experiment_result = _get_experiment_data(experiment)
    experiment_result_list = {x["Event"].replace(
        "stimuli", ""): x for x in experiment_result if re.search(r"stimuli[0-9]", x["Event"])}
    return experiment_result_list


def experiment_data_processing(experiment, subject, output):
    experiment_result_list = _get_experiment_result(experiment)
    survey_result_list = _get_survey_result(experiment)
    playlist = load_csv(playlist_path)

    experiment_data_list = []
    for key, result in experiment_result_list.items():
        result_data = next((item for item in result["Data"] if 'value' in item.keys(
        ) and _get_task_id(item["name"]) is not None), None)
        music = result_data["value"]

        current_music = [x for x in playlist if x["Track Name"]
                         in music.replace(",", "")]
        genre = current_music[0]["Genre"]
        task_id = _get_task_id(result_data["name"])

        known_or_unknown = [x for x in survey_result_list[key]["Data"] if re.search(
            r"question[0-9]{1,2}-1", x["name"]) and 'value' in x]
        if not known_or_unknown:
            transform_logger.error('Whether known is none.')
            transform_logger.debug(result_data)
            continue
        known_or_unknown = known_or_unknown[0]['value']

        attention_score = [x for x in survey_result_list[key]["Data"] if re.search(
            r"question[0-9]{1,2}-2", x["name"]) and 'value' in x]
        if not attention_score:
            transform_logger.error('Attention score is none.')
            transform_logger.debug(result_data)
            continue
        attention_score = attention_score[0]['value']

        valence_score = [x for x in survey_result_list[key]["Data"] if re.search(
            r"question[0-9]{1,2}-3", x["name"]) and 'value' in x]
        if not valence_score:
            transform_logger.error('Valence score is none.')
            transform_logger.debug(result_data)
            continue
        valence_score = valence_score[0]['value']

        arousal_score = [x for x in survey_result_list[key]["Data"] if re.search(
            r"question[0-9]{1,2}-4", x["name"]) and 'value' in x]
        if not arousal_score:
            transform_logger.error('Arousal score is none.')
            transform_logger.debug(result_data)
            continue
        arousal_score = arousal_score[0]['value']
        print(arousal_score)

        experiment_data_list.append({"subject": subject, "genre": genre, "music": music, "task": task_id, "known_or_unknown": known_or_unknown, "attention_score": attention_score, "valence_score": valence_score,
                                     "arousal_score": arousal_score})

    print(experiment_data_list)
    csv_writer(f"{output}{subject}/experiment.csv", experiment_data_list)
    result_output = f"{output}"
    return result_output


def eeg_data_processing(eeg, experiment, subject, output):
    # Read eeg data from csv
    eeg_data = load_csv(eeg)

    # Read experiment result data
    experiment_result_list = _get_experiment_result(experiment)
    survey_result_list = _get_survey_result(experiment)

    playlist = load_csv(playlist_path)

    eeg_data_list = []
    for key, result in experiment_result_list.items():
        start_time = text_to_time(experiment_result_list[key]["Time"].replace(
            'Z', '+0000'), "%Y-%m-%dT%H:%M:%S.%f%z")

        end_time = text_to_time(survey_result_list[key]["Time"].replace(
            'Z', '+0000'), "%Y-%m-%dT%H:%M:%S.%f%z")

        eeg_data_during_task = [x for x in eeg_data if text_to_time(
            x["TimeStamp"], "%Y-%m-%d %H:%M:%S.%f", 'Asia/Tokyo') > start_time and text_to_time(x["TimeStamp"], "%Y-%m-%d %H:%M:%S.%f", 'Asia/Tokyo') < end_time]
        raw_eeg_data_list = [[_convert_single(x["RAW_TP9"]), _convert_single(x["RAW_AF7"]), _convert_single(x["RAW_AF8"]), _convert_single(
            x["RAW_TP10"])] for x in eeg_data_during_task if x["RAW_TP9"] != "" or x["RAW_AF7"] != "" or x["RAW_AF8"] != "" or x["RAW_TP10"] != ""]

        tensor_eeg_data_list = torch.tensor(
            np.array(raw_eeg_data_list).T, dtype=torch.float32)

        eeg_hsi_list = [[_convert_single(x["HSI_TP9"]), _convert_single(x["HSI_AF7"]), _convert_single(x["HSI_AF8"]), _convert_single(
            x["HSI_TP10"])] for x in eeg_data_during_task if x["HSI_TP9"] != "" or x["HSI_AF7"] != "" or x["HSI_AF8"] != "" or x["HSI_TP10"] != ""]

        if _is_noisy_data(eeg_hsi_list):
            transform_logger.error('Data is noisy.')
            transform_logger.debug(result)
            continue

        result_data = next((item for item in result["Data"] if 'value' in item.keys(
        ) and _get_task_id(item["name"]) is not None), None)
        if result_data is None:
            transform_logger.debug(result)
            transform_logger.error('Data is none.')
            continue

        task_id = _get_task_id(result_data["name"])
        music = result_data["value"]
        attention_score = [x for x in survey_result_list[key]["Data"] if re.search(
            r"question[0-9]{1,2}-2", x["name"]) and 'value' in x]
        if not attention_score:
            transform_logger.error('Attention score is none.')
            transform_logger.debug(result_data)
            continue

        attention_score = attention_score[0]['value']

        current_music = [x for x in playlist if x["Track Name"]
                         in music.replace(",", "")]
        genre = current_music[0]["Genre"]
        eeg_data_list.append({"genre": genre, "data": tensor_eeg_data_list,
                             "music": music, "task": task_id, "score": attention_score})

    eeg_data_list = sorted(
        eeg_data_list, key=lambda x: (x["score"], x["genre"]))
    for eeg_data in eeg_data_list:
        base = f"{output}{subject}"
        phase = _detect_phase(eeg_data["genre"])
        output_path = f'{base}/{phase}/{eeg_data["music"]}/{eeg_data["task"]}/{eeg_data["score"]}/eeg.pkl'
        pickle_writer(output_path, eeg_data["data"])

    result_output = f"{output}"
    return result_output
