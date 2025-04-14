from config import applications, data, request
from processor import Processor

if __name__ == '__main__':
    """Entry point."""
    proc = Processor()
    output = proc.get_data(data, applications, request)
    print(output)
