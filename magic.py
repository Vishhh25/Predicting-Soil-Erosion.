from math import copysign
import fire
from pathlib import Path
import csv
from tempfile import TemporaryDirectory
from cloudpathlib import S3Client, CloudPath
from osgeo import gdal, osr


SMOL_SIZE = 9
BIGG_SIZE = 120
AWS_CREDS = {
    "aws_access_key": "DPX89LK4WYPZ5ZSK1AEK",
    "aws_secret_access_key": "AlfNegw0IOLwMhvJ1h02l2wSXQRCMcTh5s7WFecX",
    "endpoint_url": "https://eodata.dataspace.copernicus.eu",
}
BAND_IDS = ["B04", "B03", "B02"]  # RGB for now (in order)
RES = 10  # the resolution of interest for band selection


class MagicRecord:
    def __init__(self):
        self.size_in_pixels = BIGG_SIZE

    def create_record(self, file_name):
        # if json record already exists, bye
        file = Path(file_name)
        p, _, ggp = file.parents[0:3]  # fancy indexing not used: iterable
        record_dir = ggp / "records" / p.name
        if record_dir.exists():
            raise ValueError("nothing to do here; record dir exists already")
        else:
            record_dir.mkdir(parents=True, exist_ok=True)

        # get the S3 path from the file whose name is given as input
        # along with other metadata such as the targetted coordinates
        # and the C-factor which will act as label in the task at hand
        c_fac = 0.0  # to prevent unboundedness
        s3_path = ""  # same as above
        lon, lat = 0.0, 0.0
        try:
            with file.open("r", newline="", encoding="utf-8") as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in csv_reader:
                    lon, lat, _, c_fac, s3_path = row  # unpack
        except Exception as e:
            print(f"exception occurred ({e}) with file: {file_name}")
        lon, lat = float(lon), float(lat)
        # save the C-factor in a label file
        with (record_dir / "label.txt").open("w", encoding="utf-8") as text_file:
            text_file.write(str(c_fac))
        # download the S3 file to a temporary directory
        s3_client = S3Client(
            aws_access_key_id=AWS_CREDS["aws_access_key"],
            aws_secret_access_key=AWS_CREDS["aws_secret_access_key"],
            endpoint_url=AWS_CREDS["endpoint_url"],
        )
        s3_client.set_as_default_client()
        cloud_path = CloudPath(f"s3:/{s3_path}", client=s3_client)  # s3_path is a str
        print(type(cloud_path))
        with TemporaryDirectory() as tmp_dir:
            # download what's there
            cloud_path.download_to(tmp_dir)
            # go through the downloaded content
            ewalk = Path(tmp_dir).glob("**/*")
            files = [x for x in ewalk if x.is_file()]
            with (record_dir / "list_dl_s2_files.txt").open(
                "w", encoding="utf-8"
            ) as text_file:
                for f in files:
                    text_file.write(f"{str(f)}\n")
            # filter the files to only keep the bands we want
            files = [
                x
                for x in files
                if (
                    x.suffix == ".jp2"
                    and any(f"_{band_id}_" in x.stem for band_id in BAND_IDS)
                    and f"{RES}m" in x.stem
                )
            ]
            for f in files:
                print(f.name)  # for debug || sanity check
            # gdal stuffs
            for f in files:
                src = gdal.Open(str(f))
                ulx, xres, _, uly, _, yres = src.GetGeoTransform()  # 'ul': 'upper left'
                # transform; ref: https://www.perrygeo.com/python-affine-transforms.html
                # GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
                # GT(1) w-e pixel resolution / pixel width.
                # GT(2) row rotation (typically zero).
                # GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
                # GT(4) column rotation (typically zero).
                # GT(5) n-s pixel resolution / pixel height (negative value for a north-up image).
                # this is directly from https://gdal.org/tutorials/geotransforms_tut.html
                lrx = ulx + (src.RasterXSize * xres)  # 'lr': 'lower right'
                lry = uly + (src.RasterYSize * yres)

                # setup the source projection: the one in which the openned image is represented
                source = osr.SpatialReference()
                source.ImportFromWkt(src.GetProjection())

                # setup the target projection: the usual layman angle coordinate system
                target = osr.SpatialReference()
                target.ImportFromEPSG(4326)

                # create the transform from source to target projections
                transform = osr.CoordinateTransformation(source, target)

                # transform the points: upper left and lower right
                ul_lat, ul_lon, _ = transform.TransformPoint(
                    ulx, uly
                )  # looks janky: but correct!
                lr_lat, lr_lon, _ = transform.TransformPoint(
                    lrx, lry
                )  # looks janky: but correct!

                # sanity check on angle coordinates
                print(f"{ul_lon} < {lon} < {lr_lon}")
                print(f"{ul_lat} > {lat} > {lr_lat}")

                # transform everything back from angles to meters (target to source)
                transform = osr.CoordinateTransformation(target, source)  # reverse
                x, y, _ = transform.TransformPoint(lat, lon)

                # sanity check on meter coordinates
                print(f"{ulx} < {x} < {lrx}")
                print(f"{uly} > {y} > {lry}")

                # create useful value: half of the square to crop, in meters
                half_size_in_m = RES * self.size_in_pixels // 2

                # crop and write a GeoTIFF to disk
                translate_options = gdal.TranslateOptions(
                    format="GTiff",
                    projWin=[
                        x - copysign(half_size_in_m, xres),
                        y - copysign(half_size_in_m, yres),
                        x + copysign(half_size_in_m, xres),
                        y + copysign(half_size_in_m, yres),
                    ],
                )
                dst_path = str(record_dir / f"{f.stem}.tiff")
                gdal.Translate(dst_path, src, options=translate_options)
                # this returns a osgeo.gdal.Dataset object: don't care

    def test_print(self, content):
        print(content)  # fire interprets it as string by default


if __name__ == "__main__":
    fire.Fire(MagicRecord)


# CLI command to run:
# python magic.py create_record my_file_name
# or, via the more verbose
# python magic.py create_record --file_name=my_file_name

# tests to carry out
# + check that the code quits when the record already exists
# + check what happens when we aim for the edge of a tile
