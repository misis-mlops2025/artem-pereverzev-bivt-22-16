from project_name.configs import MyConfig


def train_loop(config: MyConfig):

    num_epoch = config.num_epoch

    for n in range(num_epoch):
        print("OK")