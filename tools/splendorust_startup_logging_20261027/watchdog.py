"""Explicit owning-watchdog entry point; shares the original guardian clock."""
import argparse
from .guardian import inner_main

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-sha256',required=True)
    return inner_main(parser.parse_args())

if __name__=='__main__':raise SystemExit(main())
