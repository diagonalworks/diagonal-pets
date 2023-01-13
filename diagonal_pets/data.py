import csv
import gzip

from pathlib import Path

def make_file_data(**args):
    return FileData(**args)

class FileData:

    def __init__(self, person_data_path, activity_location_data_path, activity_location_assignment_data_path, disease_outcome_data_path, model_dir, preds_format_path=None, preds_dest_path=None, household_data_path=None, residence_location_data_path=None, population_network_data_path=None):
        self.person_data_path = person_data_path
        self.activity_location_data_path = activity_location_data_path
        self.activity_location_assignment_data_path = activity_location_assignment_data_path
        self.disease_outcome_data_path = disease_outcome_data_path
        self.model_dir = model_dir
        self.preds_format_path = preds_format_path
        self.preds_dest_path = preds_dest_path

    def person(self):
        return CSV(self.person_data_path, (("pid", int),))

    def activity_locations(self):
        return CSV(self.activity_location_data_path, (("alid", int),))

    def activity_location_assignment(self):
        return CSV(self.activity_location_assignment_data_path, (("pid", int), ("lid", int)))

    def disease_outcome(self):
        return CSV(self.disease_outcome_data_path, (("day", int), ("pid", int), ("state", str)))

    def preds_format(self):
        return CSV(self.preds_format_path, (("pid", int),))

    def targets(self):
        return CSV(self.preds_format_path, (("pid", int), ("infected", lambda x: int(x) != 0)))

    def preds(self, prefix=None):
        if prefix is not None:
            i = str(self.preds_dest_path).index("_")
            path = Path(prefix + str(self.preds_dest_path)[i:])
        else:
            path = self.preds_dest_path
        return CSV(path, (("pid", int), ("score", float)))

    def model_filename(self,note=""):
        if note:
            note = "." + note
        n = self.person_data_path.name
        return self.model_dir.joinpath(Path(n[0:n.index("_")] +  "_model" + note + ".ckpt"))
    
    def preds_dest_filename(self):
        return self.preds_dest_path

class CSV:

    def __init__(self, filename, columns):
        self.filename = filename
        self.columns = columns
        self.f = None       

    def __iter__(self):
        i = CSV(self.filename, self.columns)
        if i.filename.suffix == ".gz":
            i.f = gzip.open(i.filename, "rt")
        else:
            i.f = open(i.filename, "rt")
        i.r = csv.reader(i.f)
        headers = next(i.r)
        i.indices = [(headers.index(c), t) for (c, t) in i.columns]
        return i

    def __next__(self):
        row = next(self.r)
        return [t(row[i]) for (i, t) in self.indices]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.f:
            self.f.close()
            self.f = None

