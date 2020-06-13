import simulations
import extremity_game

if __name__ == "__main__":

    # num_batches_test = 10
    num_batches_test = 5000

    # mini_batch_size_test = (2, )
    mini_batch_size_test = (128, )  # number of target to predict, number of mini-context in context

    # num_objects_test = (5, )
    num_objects_test = (5, 10, 15)
    # num_processes_test = 12  # 12

    num_processes_test = 4

    simulations.run_simulation_grid(
        "extremity_game_acl_4",
        extremity_game.make_extremity_game_simulation,
        message_sizes=(2,),
        # num_trials= 1,
        num_trials=10,
        # object_size = (3, 4, 5, 10, 15)
        object_size=(10, 15),  # number of properties
        strict_context=(True, False),
        shared_context=(True, False),
        # strict_context=(True,),
        # shared_context=(True,),
        num_objects=num_objects_test,
        mini_batch_size=mini_batch_size_test,
        num_batches=num_batches_test,
        num_processes = num_processes_test,
        loss_type = ("cross_entropy", "mse") #  "mse"  # "cross_entropy"
    )
