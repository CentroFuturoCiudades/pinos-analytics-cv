import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import csv
import os
import ast  # Import ast for safe evaluation

CSV_FILE = "../csv/area_juegos4.csv"  # Define the CSV file name

class ImagePointSelector:
    def __init__(self, ax, img_path, label, point_list):
        self.ax = ax
        self.img = mpimg.imread(img_path)
        self.ax.imshow(self.img)
        self.label = label
        self.points = point_list  # Shared list for ordered saving
        self.labels = []
        self.polygon_line = None

        # Load pre-existing points from CSV
        self.load_points_from_csv()

        # Store original axis limits for reset functionality
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()

        # Connect events
        self.cid_click = self.ax.figure.canvas.mpl_connect("button_press_event", self.onclick)

    def load_points_from_csv(self):
        """Load points from CSV if the file exists."""
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, mode="r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    coord = self.parse_coordinate(row.get("Coordinate", ""))
                    if coord:
                        self.points.append(coord)
                        self.plot_point(coord, int(row["Number"]))
            print(f"Loaded {len(self.points)} points for {self.label}")
            self.draw_polygon()

    def parse_coordinate(self, coord_string):
        """Parse coordinate string and remove quotes if necessary."""
        if coord_string:
            # Strip any extra quotes or whitespace
            coord_string = coord_string.strip('"')
            try:
                # Safely evaluate the coordinate tuple
                return ast.literal_eval(coord_string)
            except (ValueError, SyntaxError):
                print(f"Invalid coordinate format: {coord_string}")
                return None
        return None

    def plot_point(self, coord, number):
        """Plot a loaded point on the image."""
        x, y = coord
        point = self.ax.scatter(x, y, color="red")
        label = self.ax.text(x, y, str(number), color="blue", fontsize=12)
        self.labels.append((point, label))
    
    def draw_polygon(self):
        """Draw a closed polygon connecting all current points."""
        if self.polygon_line:
            self.polygon_line.remove()

        if len(self.points) >= 3:
            xs, ys = zip(*self.points)
            xs += (xs[0],)
            ys += (ys[0],)
            self.polygon_line, = self.ax.plot(xs, ys, color="lime", linewidth=2, linestyle='-', marker='o')
            plt.draw()

    def onclick(self, event):
        """Left-click to add a point, right-click to remove a point."""
        toolbar = plt.get_current_fig_manager().toolbar
        if toolbar.mode == "" and event.inaxes == self.ax:
            if event.button == 1: # Left mouse button (Add point)
                x, y = int(event.xdata), int(event.ydata)
                self.points.append((x, y))
                self.plot_point((x, y), len(self.points))
                print(f"{self.label} - Point {len(self.points)}: ({x}, {y})")
                self.draw_polygon()
                plt.draw()
                save_points_to_csv(self.points)
            elif event.button == 3:  # Right click to delete nearest point
                if self.points:
                    distances = [np.sqrt((x - event.xdata)**2 + (y - event.ydata)**2) for x, y in self.points]
                    nearest_index = np.argmin(distances)
                    self.points.pop(nearest_index)
                    point, label = self.labels.pop(nearest_index)

                    # Remove point and label from plot
                    point.remove()
                    label.remove()
                    print(f"Point {nearest_index + 1} deleted from {self.label}")
                    self.draw_polygon()
                    plt.draw()
                    save_points_to_csv(self.points)

    def reset_zoom(self, event):
        """Reset zoom to original view."""
        self.ax.set_xlim(self.original_xlim)
        self.ax.set_ylim(self.original_ylim)
        plt.draw()

def save_points_to_csv(points):
    """Save points to a CSV file with headers: Number, Birdseye, Camera."""
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Number", "Coordinate"])
        for i, coord in enumerate(points):
            writer.writerow([i + 1, coord])
    print(f"Points saved to {CSV_FILE}")

def display_image(img_path):
    global image_points
    image_points = []

    ax = plt.subplot(1, 1, 1)
    selector = ImagePointSelector(ax, img_path, "Image View", image_points)

    ax.set_title("Image View")

    plt.gcf().canvas.mpl_connect("key_press_event", lambda event: selector.reset_zoom(event) if event.key == "r" else None)
    plt.show()

if __name__ == "__main__":
    img_path = "../imgs/birdseye.jpg" 

    print("Launching window for single image.")
    print("Press 'r' to reset zoom. Left-click to add points. Right-click to remove.")
    display_image(img_path)