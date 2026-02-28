import numpy as np

def score_speed(length_ratio):
    deviation = abs(length_ratio - 1.0)
    
    if deviation <= 0.2:
        score = 100 - (deviation / 0.2) * 10
        category = "Excellent"
    elif deviation <= 0.5:
        score = 90 - ((deviation - 0.2) / 0.3) * 50
        category = "Fair"
    else:
        score = max(0, 40 - ((deviation - 0.5) / 0.5) * 40)
        category = "Critical"
    
    return {
        'score': round(score, 2),
        'normalized_score': round(score / 100, 3),
        'category': category
    }


def score_mae(mae):
    if mae <= 10:
        score = 100 - (mae / 10) * 10
        category = "Excellent"
    elif mae <= 25:
        score = 90 - ((mae - 10) / 15) * 50
        category = "Fair"
    else:
        score = max(0, 40 - ((mae - 25) / 25) * 40)
        category = "Critical"
    
    return {
        'score': round(score, 2),
        'normalized_score': round(score / 100, 3),
        'category': category
    }


def score_rmse(rmse, mae):
    if rmse <= 10:
        score = 100 - (rmse / 10) * 10
        category = "Excellent"
    elif rmse <= 25:
        score = 90 - ((rmse - 10) / 15) * 50
        category = "Fair"
    else:
        score = max(0, 40 - ((rmse - 25) / 25) * 40)
        category = "Critical"
    
    consistency_ratio = rmse / mae if mae > 0 else 1.0
    if consistency_ratio > 1.5:
        score *= 0.9
    
    return {
        'score': round(score, 2),
        'normalized_score': round(score / 100, 3),
        'category': category,
        'consistency_ratio': round(consistency_ratio, 3)
    }


def score_rom(rom_difference, trainer_rom):
    abs_diff = abs(rom_difference)
    rom_percentage = (rom_difference / trainer_rom) * 100 if trainer_rom > 0 else 0
    
    if abs_diff <= 10:
        score = 100 - (abs_diff / 10) * 10
        category = "Excellent"
    elif abs_diff <= 25:
        score = 90 - ((abs_diff - 10) / 15) * 50
        category = "Fair"
    else:
        score = max(0, 40 - ((abs_diff - 25) / 25) * 40)
        category = "Critical"
    
    return {
        'score': round(score, 2),
        'normalized_score': round(score / 100, 3),
        'category': category,
        'rom_difference_degrees': round(rom_difference, 2),
        'rom_difference_percentage': round(rom_percentage, 2)
    }


def score_max_deviation(max_deviation):
    if max_deviation <= 10:
        score = 100 - (max_deviation / 10) * 10
        category = "Excellent"
    elif max_deviation <= 30:
        score = 90 - ((max_deviation - 10) / 20) * 50
        category = "Fair"
    else:
        score = max(0, 40 - ((max_deviation - 30) / 30) * 40)
        category = "Critical"
    
    return {
        'score': round(score, 2),
        'normalized_score': round(score / 100, 3),
        'category': category
    }


def calculate_overall_score(scores_dict, weights=None):
    if weights is None:
        weights = {
            'speed': 0.20,
            'mae': 0.30,
            'rmse': 0.25,
            'rom': 0.15,
            'max_deviation': 0.10
        }
    
    overall_score = (
        scores_dict['speed']['score'] * weights['speed'] +
        scores_dict['mae']['score'] * weights['mae'] +
        scores_dict['rmse']['score'] * weights['rmse'] +
        scores_dict['rom']['score'] * weights['rom'] +
        scores_dict['max_deviation']['score'] * weights['max_deviation']
    )
    
    if overall_score >= 80:
        category = "Excellent"
    elif overall_score >= 50:
        category = "Fair"
    else:
        category = "Critical"
    
    weak_areas = []
    for metric, score_obj in scores_dict.items():
        if score_obj['score'] < 60:
            weak_areas.append({
                'metric': metric,
                'score': score_obj['score'],
                'category': score_obj['category']
            })
    weak_areas.sort(key=lambda x: x['score'])
    
    return {
        'overall_score': round(overall_score, 2),
        'normalized_score': round(overall_score / 100, 3),
        'category': category,
        'weak_areas': weak_areas[:3]
    }


def score_all_metrics(results):
    scores = {
        'speed': score_speed(results['length_ratio']),
        'mae': score_mae(results['mae']),
        'rmse': score_rmse(results['rmse'], results['mae']),
        'rom': score_rom(results['rom_difference'], results['trainer_rom']),
        'max_deviation': score_max_deviation(results['max_deviation'])
    }
    
    scores['overall'] = calculate_overall_score(scores)
    return scores


def interpolate_to_length(arr, target_length):
    current_length = len(arr)
    grid_old = np.linspace(0, 1, current_length)
    grid_new = np.linspace(0, 1, target_length)
    interpolated = np.interp(grid_new, grid_old, arr)
    return interpolated


def compare_reps(user_array, trainer_array):
    user_length = len(user_array)
    trainer_length = len(trainer_array)
    length_ratio = user_length / trainer_length if trainer_length > 0 else 0
    
    user_min = np.min(user_array)
    user_max = np.max(user_array)
    user_start = user_array[0]
    user_end = user_array[-1]
    
    trainer_min = np.min(trainer_array)
    trainer_max = np.max(trainer_array)
    trainer_start = trainer_array[0]
    trainer_end = trainer_array[-1]
    
    user_rom = user_max - user_min
    trainer_rom = trainer_max - trainer_min
    rom_difference = user_rom - trainer_rom
    
    avg_length = int(round((user_length + trainer_length) / 2))
    user_interpolated = interpolate_to_length(user_array, avg_length)
    trainer_interpolated = interpolate_to_length(trainer_array, avg_length)
    
    errors = np.abs(user_interpolated - trainer_interpolated)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((user_interpolated - trainer_interpolated) ** 2))
    max_deviation = np.max(errors)
    max_deviation_frame = np.argmax(errors)
    
    return {
        'user_length': user_length,
        'trainer_length': trainer_length,
        'length_ratio': length_ratio,
        'user_min': user_min,
        'user_max': user_max,
        'user_start': user_start,
        'user_end': user_end,
        'trainer_min': trainer_min,
        'trainer_max': trainer_max,
        'trainer_start': trainer_start,
        'trainer_end': trainer_end,
        'user_rom': user_rom,
        'trainer_rom': trainer_rom,
        'rom_difference': rom_difference,
        'mae': mae,
        'rmse': rmse,
        'max_deviation': max_deviation,
        'max_deviation_frame': max_deviation_frame
    }


def analyze_rep(user_array, trainer_array):
    results = compare_reps(user_array, trainer_array)
    scores = score_all_metrics(results)
    return scores


if __name__ == "__main__":
    trainer = np.array([30, 45, 60, 80, 95, 110, 120, 130, 135, 130, 120, 110, 95, 80, 60, 45, 30])
    user = np.array([50, 70, 85, 75, 90, 70, 85, 95, 90])    
    result = analyze_rep(user, trainer)
    print(result)