def create_exp_name(exp_name, args):

    exp_name += ('_Reward_' + args.reward)
    exp_name += ('_BatchSize' + str(args.batch_size))
    exp_name += ('_Epoch' + str(args.num_epochs))
    exp_name += ('_Width' + str(args.width))
    exp_name += ('_Dim' + str(args.desc_dim))

    exp_name += '_Debug' if args.debug else ''

    return exp_name
