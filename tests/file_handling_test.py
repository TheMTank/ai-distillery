""" Tests foro file_handling.py module """
from aidistillery.file_handling import identifier_from_path

def test_identifier_from_path():
    assert identifier_from_path("0507037v3.pdf.txt") == '0507037v3'
    assert identifier_from_path("0705.4676v8.pdf.txt") == '0705.4676v8'
    assert identifier_from_path("data/txt/0705.4676v8.pdf.txt") == '0705.4676v8'
