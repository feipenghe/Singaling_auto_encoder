import simulations
import extremity_game

if __name__ == "__main__":
    extremity_sim = extremity_game.make_extremity_game_simulation(
        object_size=3,
        num_objects=15,
        message_sizes=(2,),
        strict_context=True,
        shared_context=True,
        num_batches=100,
        mini_batch_size=128,
        num_trials=1,
    )
    simulations.run_simulation(extremity_sim, visualize=False)
