import xml.etree.ElementTree as ET
import urllib



call_keys = {
                "Called Strike": 0,
                "Ball": 1
            }

def handle_pitch(pitch):
    if pitch['des'] in call_keys:
        call = call_keys[pitch['des']]
        x = float(pitch['px'])
        y = float(pitch['pz'])
        return x, y, call


def parse(urls):
    pitches = []

    for url in urls:
        tree = ET.parse(urllib.urlopen(url))
        game = tree.getroot()

        for inning in game:
            for half in inning:
                if half.tag == "top" or half.tag == "bottom":
                    for at_bat in half:
                        if at_bat.tag == "atbat":
                            for action in at_bat:
                                if action.tag == "pitch":
                                    pitch = handle_pitch(action.attrib)
                                    if pitch is not None:
                                        pitches.append(pitch)
    return pitches
