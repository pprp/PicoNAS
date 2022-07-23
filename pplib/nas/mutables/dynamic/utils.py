def add_link(pre_channel_mutable, channel_mutable):
    channel_mutable.register_same_mutable(pre_channel_mutable)
    pre_channel_mutable.register_same_mutable(channel_mutable)
