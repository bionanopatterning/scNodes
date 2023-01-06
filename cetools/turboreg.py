

def create():
    return TurboregTool()


class TurboregTool:
    title = "TurboReg"
    description = "Register images of equal or different sizes based on image intensity.\n" \
                  "Useful to register high-magnification TEM images to low-magnification\n" \
                  "images of the same region (e.g. to register Exposure/Search frames onto\n" \
                  "Overview images.)\n" \
                  "Requires manual input of approximate\n" \
                  "location of child within parent image."

    def __init__(self):
        pass
    # TODO: how to handle input for dynamically imported ceTools()?
