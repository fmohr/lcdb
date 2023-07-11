from lcdb.workflow._util import get_schedule_for_number_of_instances


def main():
    anchors = get_schedule_for_number_of_instances(1473, 0.1, 0.1)
    print(anchors)

if __name__ == "__main__":
    main()