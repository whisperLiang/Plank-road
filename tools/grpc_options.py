DEFAULT_GRPC_MESSAGE_SIZE_MB = 512
DEFAULT_GRPC_MESSAGE_SIZE_BYTES = DEFAULT_GRPC_MESSAGE_SIZE_MB * 1024 * 1024


def grpc_message_options(max_message_length: int = DEFAULT_GRPC_MESSAGE_SIZE_BYTES):
    limit = int(max_message_length)
    return [
        ("grpc.max_send_message_length", limit),
        ("grpc.max_receive_message_length", limit),
    ]
