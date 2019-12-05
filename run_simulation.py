import simulations
import extremity_game

if __name__ == "__main__":
    simulations.run_simulation_grid(
        "extremity_game_acl_4",
        extremity_game.make_extremity_game_simulation,
        message_sizes=(2,),
        num_trials=20,
        object_size=(5,),
        strict_context=(True, False),
        shared_context=(True, False),
        num_objects=(5, 10, 15,),
        mini_batch_size=(128,),
        num_batches=(5000,),
        num_processes=12,
    )
