import csv
import sqlalchemy as sa
from dotenv import load_dotenv
import os

def read_coordinates_from_csv(csv_path):
    coords = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            coord = row["Coordinate"].strip("()")
            x, y = map(float, coord.split(","))
            coords.append((x, y))
    return coords

def build_polygon_wkt(coords):
    # Ensure polygon is closed
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    coord_strings = [f"{int(x)} {int(y)}" for x, y in coords]
    return f"POLYGON(({', '.join(coord_strings)}))"

def insert_area(engine, area_name, polygon_wkt, geometry_type="polygon"):
    conn = engine.connect()
    trans = conn.begin()
    try:
        conn.execute(sa.text("""
            INSERT INTO areasofinterest (
                area_name,
                field_geometry,
                geometry_type
            )
            VALUES (
                :area_name,
                ST_GeomFromText(:polygon_wkt, 0),
                :geometry_type
            )
        """), {
            "area_name": area_name,
            "polygon_wkt": polygon_wkt,
            "geometry_type": geometry_type
        })
        trans.commit()
    except Exception as e:
        print("Insert failed:", e)
        trans.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python insert_polygon.py <path_to_csv>")
        exit(1)
    load_dotenv()
    
    csv_path = sys.argv[1]
    area_name = csv_path.split("/")[-1].replace(".csv", "")

    coords = read_coordinates_from_csv(csv_path)
    polygon_wkt = build_polygon_wkt(coords)
    
    host = os.getenv('HOST')
    port = int(os.getenv('DB_PORT'))
    db = os.getenv('DB_NAME')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    engine = sa.create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}", echo=True)
    
    print("WKT to insert:", polygon_wkt)
    print("area name:", area_name)
    insert_area(engine, area_name, polygon_wkt)
    print("Polygon inserted successfully.")
