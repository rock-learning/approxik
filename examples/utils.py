import sys


def parse_args(base_link="kuka_lbr_l_link_0", ee_link="kuka_lbr_l_link_7"):
    args = sys.argv
    if len(args) < 2:
        print("Usage: <script> <filename> [<base_link> <endeffector_link>]")
        exit(1)
    else:
        filename = args[1]
    if len(args) >= 4:
        if args[2] != "trajectory":
            base_link = args[2]
            ee_link = args[3]
    print("URDF: '%s'" % filename)
    print("  base link: '%s'" % base_link)
    print("  end-effector link: '%s'" % ee_link)
    return filename, base_link, ee_link
