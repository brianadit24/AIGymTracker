from src import calculate_angle

pose = {
    'a': [0.6121258735656738, 0.7149084806442261],
    'b': [0.8708001375198364, 0.8325786590576172],
    'c': [0.911347508430481, 0.47694310545921326]
}

def test_calculate_angle():
    result = calculate_angle.calculate_angle(pose['a'], pose['b'], pose['c'])
    assert result == 72.04377271386271