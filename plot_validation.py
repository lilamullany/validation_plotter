import matplotlib.pyplot as plt
import glob
 

def main():
    files_to_parse = [
        "version2_output/tensorflow-logs.log",
        "version2_output/tensorflow-logs2.log"
    ]
    
    parser = Parser(files_to_parse, [], [])
    parser.update_points()
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title("Refuse Model Version 2 Loss vs. Steps")
    plt.plot(parser.steps, parser.loss)
    plt.show()

class Parser:
    def __init__(self, files_to_parse, steps=[], loss=[]):
        self.files_to_parse = files_to_parse
        self.steps = steps
        self.loss = loss

    def update_points(self):
        for filename in self.files_to_parse:
            with open(filename) as f:
                for line in f.readlines():
                    if "Loss/total_loss = " in line:
                        self.loss.append(float(line.split("Loss/total_loss = ")[-1].split(", ")[0]))
                        self.steps.append(int(line.split("global_step = ")[-1].split(", ")[0]))


if __name__ == "__main__":
    main()