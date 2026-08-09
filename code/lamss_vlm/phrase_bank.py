from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Phrase:
    identifier: int
    feature: str
    grade: int
    text: str


PHRASES: tuple[Phrase, ...] = (
    Phrase(0, "hoffa", 0, "grade 0 Hoffa synovitis: no change in the infrapatellar fat pad"),
    Phrase(1, "hoffa", 0, "Hoffa synovitis grade 0, no involvement of the infrapatellar fat pad"),
    Phrase(2, "hoffa", 0, "no Hoffa synovitis at grade 0 in the infrapatellar fat pad"),
    Phrase(3, "hoffa", 0, "infrapatellar fat pad demonstrates no Hoffa synovitis, grade 0"),
    Phrase(4, "hoffa", 0, "ordinal grade 0 Hoffa synovitis with no imaging appearance"),
    Phrase(5, "hoffa", 0, "Hoffa synovitis assessment is grade 0: no abnormality"),
    Phrase(6, "hoffa", 0, "grade 0 finding for Hoffa synovitis, no signal in the infrapatellar fat pad"),
    Phrase(7, "hoffa", 0, "infrapatellar fat pad appearance supports grade 0 Hoffa synovitis with no extent"),
    Phrase(8, "hoffa", 0, "MOAKS Hoffa synovitis grade 0, no burden"),
    Phrase(9, "hoffa", 1, "grade 1 Hoffa synovitis: mild change in the infrapatellar fat pad"),
    Phrase(10, "hoffa", 1, "Hoffa synovitis grade 1, mild involvement of the infrapatellar fat pad"),
    Phrase(11, "hoffa", 1, "mild Hoffa synovitis at grade 1 in the infrapatellar fat pad"),
    Phrase(12, "hoffa", 1, "infrapatellar fat pad demonstrates mild Hoffa synovitis, grade 1"),
    Phrase(13, "hoffa", 1, "ordinal grade 1 Hoffa synovitis with mild imaging appearance"),
    Phrase(14, "hoffa", 1, "Hoffa synovitis assessment is grade 1: mild abnormality"),
    Phrase(15, "hoffa", 1, "grade 1 finding for Hoffa synovitis, mild signal in the infrapatellar fat pad"),
    Phrase(16, "hoffa", 1, "infrapatellar fat pad appearance supports grade 1 Hoffa synovitis with mild extent"),
    Phrase(17, "hoffa", 1, "MOAKS Hoffa synovitis grade 1, mild burden"),
    Phrase(18, "hoffa", 2, "grade 2 Hoffa synovitis: moderate change in the infrapatellar fat pad"),
    Phrase(19, "hoffa", 2, "Hoffa synovitis grade 2, moderate involvement of the infrapatellar fat pad"),
    Phrase(20, "hoffa", 2, "moderate Hoffa synovitis at grade 2 in the infrapatellar fat pad"),
    Phrase(21, "hoffa", 2, "infrapatellar fat pad demonstrates moderate Hoffa synovitis, grade 2"),
    Phrase(22, "hoffa", 2, "ordinal grade 2 Hoffa synovitis with moderate imaging appearance"),
    Phrase(23, "hoffa", 2, "Hoffa synovitis assessment is grade 2: moderate abnormality"),
    Phrase(24, "hoffa", 2, "grade 2 finding for Hoffa synovitis, moderate signal in the infrapatellar fat pad"),
    Phrase(25, "hoffa", 2, "infrapatellar fat pad appearance supports grade 2 Hoffa synovitis with moderate extent"),
    Phrase(26, "hoffa", 2, "MOAKS Hoffa synovitis grade 2, moderate burden"),
    Phrase(27, "hoffa", 3, "grade 3 Hoffa synovitis: severe change in the infrapatellar fat pad"),
    Phrase(28, "hoffa", 3, "Hoffa synovitis grade 3, severe involvement of the infrapatellar fat pad"),
    Phrase(29, "hoffa", 3, "severe Hoffa synovitis at grade 3 in the infrapatellar fat pad"),
    Phrase(30, "hoffa", 3, "infrapatellar fat pad demonstrates severe Hoffa synovitis, grade 3"),
    Phrase(31, "hoffa", 3, "ordinal grade 3 Hoffa synovitis with severe imaging appearance"),
    Phrase(32, "hoffa", 3, "Hoffa synovitis assessment is grade 3: severe abnormality"),
    Phrase(33, "hoffa", 3, "grade 3 finding for Hoffa synovitis, severe signal in the infrapatellar fat pad"),
    Phrase(34, "hoffa", 3, "infrapatellar fat pad appearance supports grade 3 Hoffa synovitis with severe extent"),
    Phrase(35, "hoffa", 3, "MOAKS Hoffa synovitis grade 3, severe burden"),
    Phrase(36, "effusion", 0, "grade 0 effusion synovitis: no change in the joint cavity"),
    Phrase(37, "effusion", 0, "effusion synovitis grade 0, no involvement of the joint cavity"),
    Phrase(38, "effusion", 0, "no effusion synovitis at grade 0 in the joint cavity"),
    Phrase(39, "effusion", 0, "joint cavity demonstrates no effusion synovitis, grade 0"),
    Phrase(40, "effusion", 0, "ordinal grade 0 effusion synovitis with no imaging appearance"),
    Phrase(41, "effusion", 0, "effusion synovitis assessment is grade 0: no abnormality"),
    Phrase(42, "effusion", 0, "grade 0 finding for effusion synovitis, no signal in the joint cavity"),
    Phrase(43, "effusion", 0, "joint cavity appearance supports grade 0 effusion synovitis with no extent"),
    Phrase(44, "effusion", 0, "MOAKS effusion synovitis grade 0, no burden"),
    Phrase(45, "effusion", 1, "grade 1 effusion synovitis: mild change in the joint cavity"),
    Phrase(46, "effusion", 1, "effusion synovitis grade 1, mild involvement of the joint cavity"),
    Phrase(47, "effusion", 1, "mild effusion synovitis at grade 1 in the joint cavity"),
    Phrase(48, "effusion", 1, "joint cavity demonstrates mild effusion synovitis, grade 1"),
    Phrase(49, "effusion", 1, "ordinal grade 1 effusion synovitis with mild imaging appearance"),
    Phrase(50, "effusion", 1, "effusion synovitis assessment is grade 1: mild abnormality"),
    Phrase(51, "effusion", 1, "grade 1 finding for effusion synovitis, mild signal in the joint cavity"),
    Phrase(52, "effusion", 1, "joint cavity appearance supports grade 1 effusion synovitis with mild extent"),
    Phrase(53, "effusion", 1, "MOAKS effusion synovitis grade 1, mild burden"),
    Phrase(54, "effusion", 2, "grade 2 effusion synovitis: moderate change in the joint cavity"),
    Phrase(55, "effusion", 2, "effusion synovitis grade 2, moderate involvement of the joint cavity"),
    Phrase(56, "effusion", 2, "moderate effusion synovitis at grade 2 in the joint cavity"),
    Phrase(57, "effusion", 2, "joint cavity demonstrates moderate effusion synovitis, grade 2"),
    Phrase(58, "effusion", 2, "ordinal grade 2 effusion synovitis with moderate imaging appearance"),
    Phrase(59, "effusion", 2, "effusion synovitis assessment is grade 2: moderate abnormality"),
    Phrase(60, "effusion", 2, "grade 2 finding for effusion synovitis, moderate signal in the joint cavity"),
    Phrase(61, "effusion", 2, "joint cavity appearance supports grade 2 effusion synovitis with moderate extent"),
    Phrase(62, "effusion", 2, "MOAKS effusion synovitis grade 2, moderate burden"),
    Phrase(63, "effusion", 3, "grade 3 effusion synovitis: severe change in the joint cavity"),
    Phrase(64, "effusion", 3, "effusion synovitis grade 3, severe involvement of the joint cavity"),
    Phrase(65, "effusion", 3, "severe effusion synovitis at grade 3 in the joint cavity"),
    Phrase(66, "effusion", 3, "joint cavity demonstrates severe effusion synovitis, grade 3"),
    Phrase(67, "effusion", 3, "ordinal grade 3 effusion synovitis with severe imaging appearance"),
    Phrase(68, "effusion", 3, "effusion synovitis assessment is grade 3: severe abnormality"),
    Phrase(69, "effusion", 3, "grade 3 finding for effusion synovitis, severe signal in the joint cavity"),
    Phrase(70, "effusion", 3, "joint cavity appearance supports grade 3 effusion synovitis with severe extent"),
    Phrase(71, "effusion", 3, "MOAKS effusion synovitis grade 3, severe burden"),
    Phrase(72, "bml", 0, "grade 0 bone marrow lesion: no change in the subchondral bone"),
    Phrase(73, "bml", 0, "bone marrow lesion grade 0, no involvement of the subchondral bone"),
    Phrase(74, "bml", 0, "no bone marrow lesion at grade 0 in the subchondral bone"),
    Phrase(75, "bml", 0, "subchondral bone demonstrates no bone marrow lesion, grade 0"),
    Phrase(76, "bml", 0, "ordinal grade 0 bone marrow lesion with no imaging appearance"),
    Phrase(77, "bml", 0, "bone marrow lesion assessment is grade 0: no abnormality"),
    Phrase(78, "bml", 0, "grade 0 finding for bone marrow lesion, no signal in the subchondral bone"),
    Phrase(79, "bml", 0, "subchondral bone appearance supports grade 0 bone marrow lesion with no extent"),
    Phrase(80, "bml", 0, "MOAKS bone marrow lesion grade 0, no burden"),
    Phrase(81, "bml", 1, "grade 1 bone marrow lesion: mild change in the subchondral bone"),
    Phrase(82, "bml", 1, "bone marrow lesion grade 1, mild involvement of the subchondral bone"),
    Phrase(83, "bml", 1, "mild bone marrow lesion at grade 1 in the subchondral bone"),
    Phrase(84, "bml", 1, "subchondral bone demonstrates mild bone marrow lesion, grade 1"),
    Phrase(85, "bml", 1, "ordinal grade 1 bone marrow lesion with mild imaging appearance"),
    Phrase(86, "bml", 1, "bone marrow lesion assessment is grade 1: mild abnormality"),
    Phrase(87, "bml", 1, "grade 1 finding for bone marrow lesion, mild signal in the subchondral bone"),
    Phrase(88, "bml", 1, "subchondral bone appearance supports grade 1 bone marrow lesion with mild extent"),
    Phrase(89, "bml", 1, "MOAKS bone marrow lesion grade 1, mild burden"),
    Phrase(90, "bml", 2, "grade 2 bone marrow lesion: moderate change in the subchondral bone"),
    Phrase(91, "bml", 2, "bone marrow lesion grade 2, moderate involvement of the subchondral bone"),
    Phrase(92, "bml", 2, "moderate bone marrow lesion at grade 2 in the subchondral bone"),
    Phrase(93, "bml", 2, "subchondral bone demonstrates moderate bone marrow lesion, grade 2"),
    Phrase(94, "bml", 2, "ordinal grade 2 bone marrow lesion with moderate imaging appearance"),
    Phrase(95, "bml", 2, "bone marrow lesion assessment is grade 2: moderate abnormality"),
    Phrase(96, "bml", 2, "grade 2 finding for bone marrow lesion, moderate signal in the subchondral bone"),
    Phrase(97, "bml", 2, "subchondral bone appearance supports grade 2 bone marrow lesion with moderate extent"),
    Phrase(98, "bml", 2, "MOAKS bone marrow lesion grade 2, moderate burden"),
    Phrase(99, "bml", 3, "grade 3 bone marrow lesion: severe change in the subchondral bone"),
)


def select_phrases(epoch: int) -> tuple[Phrase, ...]:
    size = 10 if epoch < 30 else 25 if epoch < 60 else 50
    return PHRASES[:size]


def phrases_for(feature: str, grade: int | None = None) -> tuple[Phrase, ...]:
    return tuple(item for item in PHRASES if item.feature == feature and (grade is None or item.grade == grade))


def validate_phrase_bank(items: Iterable[Phrase]) -> None:
    values = tuple(items)
    identifiers = {item.identifier for item in values}
    if len(identifiers) != len(values):
        raise ValueError("phrase identifiers must be unique")
    if any(item.grade not in {0, 1, 2, 3} for item in values):
        raise ValueError("phrase grades must be between zero and three")
    if any(item.feature not in {"hoffa", "effusion", "bml"} for item in values):
        raise ValueError("phrase feature is unknown")

