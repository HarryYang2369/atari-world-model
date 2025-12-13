# train_controller_breakout.py
# Evolution Strategies training for the Breakout controller
#
# Usage (typical):
#   mpirun -np 9 python -u train_controller_breakout.py -o pepg -n 8
#
# After some training, you will get a file like:
#   log/breakout.pepg.16.8.json
# which you can load with:
#   python model_breakout.py render log/breakout.pepg.16.8.json

from mpi4py import MPI
import numpy as np
import json
import os
import subprocess
import sys
import argparse
import time

from model import make_model, simulate
from es import CMAES, SimpleGA, OpenES, PEPG  # ES implementations

# -------------------------------------------------------------------
# ES / training settings
# -------------------------------------------------------------------
num_episode = 4          # episodes per evaluation of a controller
eval_steps = 10            # evaluate on full episodes every N optimization steps
retrain_mode = True       # if eval gets worse, reset ES mean to previous best
cap_time_mode = False     # cap max_len ~ 2x avg episode length

num_worker = 8            # MPI workers (not counting rank 0)
num_worker_trial = 8

population = num_worker * num_worker_trial

gamename = "breakout"     # used in log filenames
optimizer = "cma"         # pepg, cma, ga, ses, openes
antithetic = True         # antithetic sampling (where supported)
batch_mode = "mean"       # aggregate over episodes: 'mean' or 'min'

# seed for reproducibility
seed_start = 1

# name prefix for output files (will be set in initialize_settings)
filebase = None

# Build controller + world-model once per process
model = make_model(load_model=True)
num_params = model.param_count
es = None

# -------------------------------------------------------------------
# MPI globals
# -------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

PRECISION = 10000
SOLUTION_PACKET_SIZE = None
RESULT_PACKET_SIZE = None


# -------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------
def initialize_settings(sigma_init=0.1, sigma_decay=0.9999):
    global population, filebase, model, num_params, es
    global PRECISION, SOLUTION_PACKET_SIZE, RESULT_PACKET_SIZE
    global gamename

    population = num_worker * num_worker_trial
    os.makedirs("log", exist_ok=True)
    filebase = f"log/{gamename}.{optimizer}.{num_episode}.{population}"

    print("size of controller params:", num_params)

    # Choose optimizer
    if optimizer == "ses":
        es_local = PEPG(
            num_params,
            sigma_init=sigma_init,
            sigma_decay=sigma_decay,
            sigma_alpha=0.2,
            sigma_limit=0.02,
            elite_ratio=0.1,
            weight_decay=0.005,
            popsize=population,
        )
    elif optimizer == "ga":
        es_local = SimpleGA(
            num_params,
            sigma_init=sigma_init,
            sigma_decay=sigma_decay,
            sigma_limit=0.02,
            elite_ratio=0.1,
            weight_decay=0.005,
            popsize=population,
        )
    elif optimizer == "cma":
        es_local = CMAES(
            num_params,
            sigma_init=sigma_init,
            popsize=population,
        )
    elif optimizer == "pepg":
        es_local = PEPG(
            num_params,
            sigma_init=sigma_init,
            sigma_decay=sigma_decay,
            sigma_alpha=0.20,
            sigma_limit=0.02,
            learning_rate=0.01,
            learning_rate_decay=1.0,
            learning_rate_limit=0.01,
            weight_decay=0.005,
            popsize=population,
        )
    else:
        es_local = OpenES(
            num_params,
            sigma_init=sigma_init,
            sigma_decay=sigma_decay,
            sigma_limit=0.02,
            learning_rate=0.01,
            learning_rate_decay=1.0,
            learning_rate_limit=0.01,
            antithetic=antithetic,
            weight_decay=0.005,
            popsize=population,
        )

    es = es_local

    # recompute packet sizes now that num_params is known
    PRECISION = 10000
    SOLUTION_PACKET_SIZE = (5 + num_params) * num_worker_trial
    RESULT_PACKET_SIZE = 4 * num_worker_trial


# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------
def sprint(*args):
    print(*args)
    sys.stdout.flush()


class Seeder:
    def __init__(self, init_seed=0):
        np.random.seed(init_seed)
        self.limit = np.int32(2**31 - 1)

    def next_seed(self):
        return np.random.randint(self.limit)

    def next_batch(self, batch_size):
        return np.random.randint(self.limit, size=batch_size).tolist()


# -------------------------------------------------------------------
# Encoding / decoding packets
# -------------------------------------------------------------------
def encode_solution_packets(seeds, solutions, train_mode=1, max_len=-1):
    n = len(seeds)
    result = []
    for i in range(n):
        worker_num_local = int(i / num_worker_trial) + 1
        result.append([worker_num_local, i, seeds[i], train_mode, max_len])
        result.append(np.round(np.array(solutions[i]) * PRECISION, 0))
    result = np.concatenate(result).astype(np.int32)
    # split into num_worker equal chunks, one per slave
    result = np.split(result, num_worker)
    return result


def decode_solution_packet(packet):
    packets = np.split(packet, num_worker_trial)
    result = []
    for p in packets:
        result.append(
            [
                p[0],
                p[1],
                p[2],
                p[3],
                p[4],
                p[5:].astype(float) / PRECISION,
            ]
        )
    return result


def encode_result_packet(results):
    r = np.array(results)
    r[:, 2:4] *= PRECISION
    return r.flatten().astype(np.int32)


def decode_result_packet(packet):
    r = packet.reshape(num_worker_trial, 4)
    workers = r[:, 0].tolist()
    jobs = r[:, 1].tolist()
    fits = (r[:, 2].astype(float) / PRECISION).tolist()
    times = (r[:, 3].astype(float) / PRECISION).tolist()
    result = []
    n = len(jobs)
    for i in range(n):
        result.append([workers[i], jobs[i], fits[i], times[i]])
    return result


# -------------------------------------------------------------------
# Evaluation on a single worker
# -------------------------------------------------------------------
def worker(weights, seed, train_mode_int=1, max_len=-1):
    train_mode = (train_mode_int == 1)
    model.set_model_params(weights)
    reward_list, t_list = simulate(
        model,
        train_mode=train_mode,
        render_mode=False,
        num_episode=num_episode,
        seed=seed,
        max_len=max_len,
    )
    if batch_mode == "min":
        reward = float(np.min(reward_list))
    else:
        reward = float(np.mean(reward_list))
    t = float(np.mean(t_list))
    return reward, t


# -------------------------------------------------------------------
# Slave loop (MPI rank > 0)
# -------------------------------------------------------------------
def slave():
    model.make_env()
    packet = np.empty(SOLUTION_PACKET_SIZE, dtype=np.int32)
    while True:
        comm.Recv(packet, source=0)
        assert len(packet) == SOLUTION_PACKET_SIZE
        solutions = decode_solution_packet(packet)
        results = []
        for solution in solutions:
            worker_id, jobidx, seed, train_mode, max_len, weights = solution
            assert (train_mode == 1 or train_mode == 0), str(train_mode)
            worker_id = int(worker_id)
            possible_error = f"work_id = {worker_id} rank = {rank}"
            assert worker_id == rank, possible_error
            jobidx = int(jobidx)
            seed = int(seed)
            fitness, timesteps = worker(weights, seed, train_mode, max_len)
            results.append([worker_id, jobidx, fitness, timesteps])
        result_packet = encode_result_packet(results)
        assert len(result_packet) == RESULT_PACKET_SIZE
        comm.Send(result_packet, dest=0)


def send_packets_to_slaves(packet_list):
    world_size = comm.Get_size()
    assert len(packet_list) == world_size - 1
    for i in range(1, world_size):
        packet = packet_list[i - 1]
        assert len(packet) == SOLUTION_PACKET_SIZE
        comm.Send(packet, dest=i)


def receive_packets_from_slaves():
    result_packet = np.empty(RESULT_PACKET_SIZE, dtype=np.int32)

    reward_list_total = np.zeros((population, 2))
    check_results = np.ones(population, dtype=int)

    for i in range(1, num_worker + 1):
        comm.Recv(result_packet, source=i)
        results = decode_result_packet(result_packet)
        for result in results:
            worker_id = int(result[0])
            possible_error = f"work_id = {worker_id} source = {i}"
            assert worker_id == i, possible_error
            idx = int(result[1])
            reward_list_total[idx, 0] = result[2]
            reward_list_total[idx, 1] = result[3]
            check_results[idx] = 0

    check_sum = check_results.sum()
    assert check_sum == 0, check_sum
    return reward_list_total


def evaluate_batch(model_params, max_len=-1):
    # duplicate model_params across population
    solutions = [np.copy(model_params) for _ in range(es.popsize)]
    seeds = np.arange(es.popsize)

    packet_list = encode_solution_packets(
        seeds, solutions, train_mode=0, max_len=max_len
    )

    send_packets_to_slaves(packet_list)
    reward_list_total = receive_packets_from_slaves()
    reward_list = reward_list_total[:, 0]
    return float(np.mean(reward_list))


# -------------------------------------------------------------------
# Master training loop (rank 0)
# -------------------------------------------------------------------
def master():
    start_time = int(time.time())
    sprint("training", gamename)
    sprint("population", es.popsize)
    sprint("num_worker", num_worker)
    sprint("num_worker_trial", num_worker_trial)
    sys.stdout.flush()

    seeder = Seeder(seed_start)

    filename = filebase + ".json"          # current ES mean params
    filename_log = filebase + ".log.json"  # eval log
    filename_hist = filebase + ".hist.json"
    filename_hist_best = filebase + ".hist_best.json"
    filename_best = filebase + ".best.json"  # [best_params, best_eval_reward]

    model.make_env()

    t = 0
    history = []
    history_best = []
    eval_log = []
    best_reward_eval = 0.0
    best_model_params_eval = None

    max_len = -1  # no cap initially

    while True:
        t += 1

        solutions = es.ask()

        if antithetic:
            seeds = seeder.next_batch(int(es.popsize / 2))
            seeds = seeds + seeds
        else:
            seeds = seeder.next_batch(es.popsize)

        packet_list = encode_solution_packets(seeds, solutions, max_len=max_len)
        send_packets_to_slaves(packet_list)
        reward_list_total = receive_packets_from_slaves()

        reward_list = reward_list_total[:, 0]

        mean_time_step = int(np.mean(reward_list_total[:, 1]) * 100) / 100.0
        max_time_step = int(np.max(reward_list_total[:, 1]) * 100) / 100.0
        avg_reward = int(np.mean(reward_list) * 100) / 100.0
        std_reward = int(np.std(reward_list) * 100) / 100.0

        es.tell(reward_list)

        es_solution = es.result()
        model_params = es_solution[0]  # best historical solution
        reward = es_solution[1]        # best historical reward
        curr_reward = es_solution[2]   # best of current batch
        model.set_model_params(np.array(model_params).round(4))

        r_max = int(np.max(reward_list) * 100) / 100.0
        r_min = int(np.min(reward_list) * 100) / 100.0

        curr_time = int(time.time()) - start_time

        h = (
            t,
            curr_time,
            avg_reward,
            r_min,
            r_max,
            std_reward,
            int(es.rms_stdev() * 100000) / 100000.0,
            mean_time_step + 1.0,
            int(max_time_step) + 1,
        )

        if cap_time_mode:
            max_len = 2 * int(mean_time_step + 1.0)
        else:
            max_len = -1

        history.append(h)

        # Save current ES-mean params (what model_breakout.py will load)
        with open(filename, "wt") as out:
            json.dump(
                [np.array(es.current_param()).round(4).tolist()],
                out,
                sort_keys=True,
                indent=2,
                separators=(",", ": "),
            )

        with open(filename_hist, "wt") as out:
            json.dump(history, out, sort_keys=False, indent=0, separators=(",", ":"))

        sprint(gamename, h)

        if t == 1:
            best_reward_eval = avg_reward

        # Periodic full evaluation
        if t % eval_steps == 0:
            prev_best_reward_eval = best_reward_eval
            model_params_quantized = np.array(es.current_param()).round(4)
            reward_eval = evaluate_batch(model_params_quantized, max_len=-1)
            model_params_quantized = model_params_quantized.tolist()
            improvement = reward_eval - best_reward_eval

            eval_log.append([t, reward_eval, model_params_quantized])
            with open(filename_log, "wt") as out:
                json.dump(eval_log, out)

            if len(eval_log) == 1 or reward_eval > best_reward_eval:
                best_reward_eval = reward_eval
                best_model_params_eval = model_params_quantized
            else:
                if retrain_mode:
                    sprint(
                        "reset to previous best params, where best_reward_eval =",
                        best_reward_eval,
                    )
                    es.set_mu(best_model_params_eval)

            with open(filename_best, "wt") as out:
                json.dump(
                    [best_model_params_eval, best_reward_eval],
                    out,
                    sort_keys=True,
                    indent=0,
                    separators=(",", ":"),
                )

            curr_time = int(time.time()) - start_time
            best_record = [
                t,
                curr_time,
                "improvement",
                improvement,
                "curr",
                reward_eval,
                "prev",
                prev_best_reward_eval,
                "best",
                best_reward_eval,
            ]
            history_best.append(best_record)
            with open(filename_hist_best, "wt") as out:
                json.dump(
                    history_best, out, sort_keys=False, indent=0, separators=(",", ":")
                )

            sprint(
                "Eval",
                t,
                curr_time,
                "improvement",
                improvement,
                "curr",
                reward_eval,
                "prev",
                prev_best_reward_eval,
                "best",
                best_reward_eval,
            )


# -------------------------------------------------------------------
# CLI + MPI fork
# -------------------------------------------------------------------
def main(args):
    global optimizer, num_episode, eval_steps, num_worker, num_worker_trial
    global antithetic, seed_start, retrain_mode, cap_time_mode

    optimizer = args.optimizer
    num_episode = args.num_episode
    eval_steps = args.eval_steps
    num_worker = args.num_worker
    num_worker_trial = args.num_worker_trial
    antithetic = (args.antithetic == 1)
    retrain_mode = (args.retrain == 1)
    cap_time_mode = (args.cap_time == 1)
    seed_start = args.seed_start

    initialize_settings(args.sigma_init, args.sigma_decay)

    sprint("process", rank, "out of total", comm.Get_size(), "started")
    if rank == 0:
        master()
    else:
        slave()


def mpi_fork(n):
    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1",
        )
        cmd = [
            "mpiexec",
            "--map-by", ":OVERSUBSCRIBE",
            "-n", str(n),
            sys.executable,
            "-u",
        ] + sys.argv
        print(cmd)
        subprocess.check_call(cmd, env=env)
        return "parent"
    else:
        return "child"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train Breakout controller with ES (pepg, ses, openes, ga, cma)"
        )
    )

    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        help="ses, pepg, openes, ga, cma.",
        default="cma",
    )
    parser.add_argument(
        "--num_episode",
        type=int,
        default=16,
        help="num episodes per trial",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=25,
        help="evaluate every eval_steps step",
    )
    parser.add_argument("-n", "--num_worker", type=int, default=8)
    parser.add_argument(
        "-t",
        "--num_worker_trial",
        type=int,
        help="trials per worker",
        default=8,
    )
    parser.add_argument(
        "--antithetic",
        type=int,
        default=1,
        help="set to 0 to disable antithetic sampling",
    )
    parser.add_argument(
        "--cap_time",
        type=int,
        default=0,
        help="0 = disable capping timesteps to 2x average.",
    )
    parser.add_argument(
        "--retrain",
        type=int,
        default=0,
        help="0 = disable retraining if eval gets worse (only ses/openes/pepg).",
    )
    parser.add_argument(
        "-s",
        "--seed_start",
        type=int,
        default=0,
        help="initial seed",
    )
    parser.add_argument(
        "--sigma_init",
        type=float,
        default=0.1,
        help="sigma_init",
    )
    parser.add_argument(
        "--sigma_decay",
        type=float,
        default=0.999,
        help="sigma_decay",
    )

    args = parser.parse_args()
    if "parent" == mpi_fork(args.num_worker + 1):
        sys.exit(0)
    main(args)
