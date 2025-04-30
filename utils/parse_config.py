import argparse

def parse_config(parser, config, exclude=None):
    exclude = set(exclude or [])
    for key, value in config.items():
        if key not in exclude:
            parser.add_argument(
                f"--{key}",
                type=type(value),
                default=value,
                help=f"{key} (default: {value})"
            )
    return parser