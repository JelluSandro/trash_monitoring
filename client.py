import base64


def get_b64f(im_path):
    with open(im_path, 'rb') as f:
        bin_d = f.read()
        b64f = base64.b64encode(bin_d).decode('utf-8')
    return b64f

