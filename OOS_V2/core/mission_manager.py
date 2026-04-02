from physics.collision import compute_collision
from core.mission import Mission
from config import PC_THRESHOLD

def generate_missions(sats):

    missions = []

    for i in range(len(sats)):
        for j in range(i+1, len(sats)):

            pc, dist = compute_collision(sats[i], sats[j])

            if pc > PC_THRESHOLD or dist < 100:

                missions.append(
                    Mission(
                        target=sats[i],
                        pc=pc,
                        distance=dist
                    )
                )

    return missions