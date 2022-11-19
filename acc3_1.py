import csv
import argparse

def read_csv(path):
    label = []
    filename = []
    with open(f'{path}', newline='') as csvfile:

        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        # 以迴圈輸出每一列
        for i, row in enumerate(rows):
            if i==0: 
                continue
            f, l = row
            label.append(l)
            filename.append(f)

    return label, filename

if __name__ == '__main__':

    # csv_path = "./output_p1/pred.csv"
    parser = argparse.ArgumentParser(description="p 3-1",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv_path", help="csv_path", default = "./hw3/output_p1/pred.csv")
    args = parser.parse_args()
    csv_path = args.csv_path

    label, filename = read_csv(csv_path)
    total = len(label) - 1
    hit = 0
    for i in range(len(label)):
        if i == 0:
            continue
        gth = filename[i].split("_")[0]
        # print(gth)
        if gth == label[i]:
            hit += 1

    print(hit/total)
    print(hit,"/",total)
