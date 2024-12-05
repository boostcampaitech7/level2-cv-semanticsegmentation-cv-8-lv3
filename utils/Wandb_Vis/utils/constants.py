# 클래스 및 그룹 정의
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS_GROUPS = [
    [1, 4, 8, 12, 16, 20, 22, 26],
    [2, 5, 9, 13, 17, 23, 24, 29],
    [3, 6, 10, 14, 18, 21, 27, 28],
    [11, 19, 25],
    [7, 15]
]

CLASS_GROUP_LABEL = [
    'Trapezium, Capitate, Triquetrum',
    'Hamate, Scaphoid, Ulna',
    'Trapezoid, Pisiform, Radius',
    '11, 19, Lunate',
    '7, 15'
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}
