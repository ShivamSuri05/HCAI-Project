from django.http import JsonResponse
from .utils import generate_filtered_trajectory_pairs, sort_pairs_preserving_organic_diff, apply_last_action_and_get_final_grid
from .utils import decode_state_tensor, convert_to_pairs, collect_preferences, bradley_terry, compute_mean_skills, learned_reward
from django.shortcuts import render
import random
import json
from django.views.decorators.csrf import csrf_exempt
from .models import TrajectoryPreference

from .enhanced_reinforce import train_with_penalty, generate_trajectory

training_logs = []

def index(request):
    training_logs[:] = []
    TrajectoryPreference.objects.all().delete()
    return render(request, 'project5/index.html')

@csrf_exempt
def get_training_logs(request):
    parsed_logs = []
    for log in training_logs:
        # Split and parse each part
        parts = log.split('|')
        batch = parts[0].strip()
        avg_loss = parts[1].split(':')[1].strip()
        success_rate = parts[2].split(':')[1].strip()
        baseline = parts[3].split(':')[1].strip()
        kl = parts[4].split(':')[1].strip()
        parsed_logs.append({
            'batch': batch,
            'avg_loss': avg_loss,
            'success_rate': success_rate,
            'baseline': baseline,
            'kl': kl,
        })
    return JsonResponse({'logs': parsed_logs})

@csrf_exempt
def save_feedback(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            TrajectoryPreference.objects.create(
                trajectory1=data['trajectory1'],
                trajectory2=data['trajectory2'],
                choice=data.get('choice')
            )

            return JsonResponse({'status': 'success'})
        except Exception as e:
            print("Exception in save_feedback:", e)
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    else:
        print("Invalid HTTP method received:", request.method)
        return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)

@csrf_exempt
def save_feedback_final(request):
    if request.method == 'POST':
        try:
            training_logs[:] = []
            data = json.loads(request.body)

            TrajectoryPreference.objects.create(
                trajectory1=data['trajectory1'],
                trajectory2=data['trajectory2'],
                choice=data.get('choice')
            )

            all_feedback = list(TrajectoryPreference.objects.values())

            trajectory_pairs = convert_to_pairs(all_feedback)

            preferences, traj_id_map, sc_list, oc_list = collect_preferences(trajectory_pairs)

            n = len(traj_id_map)
            theta = bradley_terry(preferences, n)

            inv_traj_map = {v: k for k, v in traj_id_map.items()}

            mean_sc_skill, mean_oc_skill = compute_mean_skills(theta, sc_list, oc_list)

            train_with_penalty(mean_sc_skill, mean_oc_skill, training_logs, num_batches=200, batch_size=10)

            return JsonResponse({'status': 'success'})
        except Exception as e:
            print("Exception in save_feedback:", e)
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    else:
        print("Invalid HTTP method received:", request.method)
        return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)


def process_traj(traj):
    frames = []
    for state_tensor in traj['states']:
        grid = decode_state_tensor(state_tensor)
        frames.append(grid.tolist())  # grid is a 2D list
    final_grid = apply_last_action_and_get_final_grid(traj)
    frames.append(final_grid.tolist())
    return {
        "frames": frames,
        "organic_cheese_count": traj["organic_cheese_count"]
    }

def get_trajectory_pair(request):
    num_pairs = 20
    trajectory_pairs = generate_filtered_trajectory_pairs(num_pairs)
    trajectory_pairs = sort_pairs_preserving_organic_diff(trajectory_pairs)

    traj1, traj2 = random.choice(trajectory_pairs[2:8])

    return JsonResponse({
        "trajectory1": process_traj(traj1),
        "trajectory2": process_traj(traj2)
    })

def get_trajectory(request):
    traj = generate_trajectory()
    return JsonResponse({
        "trajectory": process_traj(traj)
    })
