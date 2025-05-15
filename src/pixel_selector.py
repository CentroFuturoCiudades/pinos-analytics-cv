import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import csv
import os
import ast  # Import ast for safe evaluation

CSV_FILE = "../csv/points_cam07.csv"  # Define the CSV file name
#CSV_FILE = "../csv/arbitrary_testing/arb_points_cam07.csv"  # Define the CSV file name
IMG2 = "../imgs/cam07.jpg"  # Update with image from camera

class ImagePointSelector:
    def __init__(self, ax, img_path, label, point_list):
        self.ax = ax
        self.img = mpimg.imread(img_path)
        self.ax.imshow(self.img)
        self.label = label
        self.points = point_list  # Shared list for ordered saving
        self.labels = []

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
                    number = int(row["Number"])
                    
                    # Safely parse Birdseye and Camera coordinates
                    birdseye_coord = self.parse_coordinate(row["Birdseye"])
                    camera_coord = self.parse_coordinate(row["Camera"])

                    if self.label == "Birdseye View" and birdseye_coord:
                        self.points.append(birdseye_coord)
                        self.plot_point(birdseye_coord, number)
                    elif self.label == "Camera View" and camera_coord:
                        self.points.append(camera_coord)
                        self.plot_point(camera_coord, number)

            print(f"Loaded {len(self.points)} points for {self.label}")

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

    def onclick(self, event):
        """Left-click to add a point, right-click to remove a point."""
        toolbar = plt.get_current_fig_manager().toolbar
        active_tool = toolbar.mode  # Get the currently active tool

        if active_tool == "":  # Only allow point selection when no tool is active
            if event.inaxes == self.ax:
                if event.button == 1:  # Left mouse button (Add point)
                    x, y = int(event.xdata), int(event.ydata)
                    self.points.append((x, y))
                    self.plot_point((x, y), len(self.points))
                    plt.draw()
                    print(f"Image {self.label} - Point {len(self.points)}: ({x}, {y})")
                    save_points_to_csv()
                elif event.button == 3:  # Right mouse button (Remove nearest point)
                    if self.points:
                        distances = [np.sqrt((x - event.xdata)**2 + (y - event.ydata)**2) for x, y in self.points]
                        nearest_index = np.argmin(distances)
                        self.points.pop(nearest_index)

                        # Remove point and label from plot
                        point, label = self.labels.pop(nearest_index)
                        point.remove()
                        label.remove()
                        plt.draw()
                        print(f"Point {nearest_index + 1} deleted from {self.label}")
                        save_points_to_csv()
        else:
            print(f"Point selection disabled due to active tool: {active_tool}")

    def reset_zoom(self, event):
        """Reset zoom to original view."""
        self.ax.set_xlim(self.original_xlim)
        self.ax.set_ylim(self.original_ylim)
        plt.draw()

def save_points_to_csv():
    """Save points to a CSV file with headers: Number, Birdseye, Camera."""
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Number", "Birdseye", "Camera"])  # Headers

        # Ensure points are saved in order
        for i in range(max(len(birdseye_points), len(camera_points))):
            birdseye_coord = birdseye_points[i] if i < len(birdseye_points) else ""
            camera_coord = camera_points[i] if i < len(camera_points) else ""
            writer.writerow([i + 1, birdseye_coord, camera_coord])

    print(f"Points saved to {CSV_FILE}")

def display_images(img1_path, img2_path):
    global birdseye_points, camera_points
    birdseye_points = []
    camera_points = []

    axs = [plt.subplot(1, 2, i+1) for i in range(2)]  # Create two independent axes

    img1_selector = ImagePointSelector(axs[0], img1_path, "Birdseye View", birdseye_points)
    img2_selector = ImagePointSelector(axs[1], img2_path, "Camera View", camera_points)

    axs[0].set_title("Birdseye View")
    axs[1].set_title("Camera View")

    # Bind reset zoom shortcut ('r' key)
    plt.gcf().canvas.mpl_connect("key_press_event", lambda event: [img1_selector.reset_zoom(event), img2_selector.reset_zoom(event)] if event.key == "r" else None)

    plt.show()

if __name__ == "__main__":
    img1_path = "../imgs/birdseye.jpg"
    img2_path = IMG2

    print("Launching separate window.")
    print("Press 'r' to reset view. Left-click to add points. Right-click to remove a point.")
    print("Selection is enabled unless zoom/pan is chosen. If no tool is active, selection is re-enabled.")
    print("Click 'q' to exit.")
    display_images(img1_path, img2_path)
