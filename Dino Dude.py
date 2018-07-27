import random
import numpy as np
import tensorflow as tf
import pygame, sys, os


class Player:

  y = 0
  y_vel = 0

  def update(self):
    self.y += self.y_vel
    if self.y < 0:
      self.y = 0
      self.y_vel = 0
    else:
      self.y_vel -= .5

  def jump(self):
    if self.y == 0:
      self.y_vel = 11

  def hit_cacti(self, cacti):
    for cactus in cacti:
      if 100 - cactus.width < cactus.x < 120 and self.y < 20:
        return True
    return False

  def render(self):
    pygame.draw.rect(pygame.display.get_surface(), (255, 255, 255), (100, pygame.display.get_surface().get_height() - 60 - self.y, 20, 40))


def init_tensor_flow():
  model = tf.keras.models.Sequential()

  model.add(tf.keras.layers.Dense(32, activation='relu'))
  model.add(tf.keras.layers.Dense(32, activation='relu'))
  model.add(tf.keras.layers.Dense(2, activation='softmax'))

  # compile the net (not training; like setting up)

  model.compile(optimizer=tf.train.AdamOptimizer(),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=[tf.keras.metrics.categorical_crossentropy])

  return model


class Cactus:

  x = 800

  def __init__(self):
    self.width = random.randint(60, 100)

  def update(self, time):
    self.x -= time / 1000.0
    return self.x < 0

  def render(self):
    pygame.draw.rect(pygame.display.get_surface(), (255, 255, 255), (self.x, pygame.display.get_surface().get_height() - 60, self.width, 40))


def gen_save_data(is_jumping, player, cacti):
  data = [float(player.y) / 200.0]
  for i in range(100):
    data.append(0)
  for cactus in cacti:
    data[int(cactus.x / 8)] = 1

  label = [0, 0]
  if is_jumping:
    label[0] = 1
    label[1] = 0
  else:
    label[0] = 0
    label[1] = 1

  return data, label


def save_data(data, labels):
  pass


def gen_live_data(cacti, player): # for model.predict(x)
  data = [float(player.y) / 200.0]
  for i in range(100):
    data.append(0)
  for cactus in cacti:
    data[int(cactus.x / 8)] = 1

  return np.array([data])


def mainAI():
  pygame.init()
  pygame.font.init()

  font = pygame.font.SysFont('Comic Sans MS', 200)
  game_over_surface = font.render('Game Over', True, (255, 128, 128))
  font = pygame.font.SysFont('Comic Sans MS', 100)
  game_over_surface_2 = font.render('Press any key to restart', True, (255, 128, 128))

  screen = pygame.display.set_mode((800, 600))
  pygame.display.set_caption('Dino Dude')

  clock = pygame.time.Clock()

  model = init_tensor_flow()
  game_over = False
  all_data = []    # save data
  all_labels = []  # save labels
  while True:
    data = []
    labels = []
    for i in range(50):
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          save_data(all_data, all_labels)
          return
      little_data, little_labels = gen_data_and_play_game(model)
      data.extend(little_data)
      labels.extend(little_labels)
      print("Completed game " + str(i))
    all_data.extend(data)
    all_labels.extend(labels)
    np_data   = np.array(all_data)
    np_labels = np.array(all_labels)
    for i in range(50):
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          save_data(all_data, all_labels)
          return
      disablePrint()
      hist = model.fit(np_data, np_labels, batch_size=32, epochs=1)
      enablePrint()
      i += 1
      print("Fitted model " + str(i) + "/50. Loss: " + str(hist.history['loss']))

    if play_game(model, screen):
      save_data(all_data, all_labels)
      return


def play_game(model, screen):  # returns need_save_and_quit

  game_over = False
  player = Player()
  cacti = []
  last_cacti = 0
  time = 4000
  data = []  # this game data
  labels = []
  clock = pygame.time.Clock()
  while not game_over:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return True
    if random.randint(0, 30) == 0 and last_cacti > 100:
      last_cacti = 0
      cacti.append(Cactus())
    if player.hit_cacti(cacti):
      print("Died")
      break
    jumping = False
    if model != None:
      jumping_data = model.predict(gen_live_data(cacti, player))[0]
      if jumping_data[0] > jumping_data[1]:
        player.jump()
        jumping = True
    else:
      if pygame.key.get_pressed()[pygame.K_SPACE]:
        player.jump()
        jumping = True

    screen.fill((0, 0, 0))

    player.update()
    player.render()

    i = 0
    while i < len(cacti):
      cactus = cacti[i]
      if cactus.update(time):  # returns is_ded (ded if offscreen)
        del (cacti[i])
        continue
      cactus.render()
      i += 1

    little_data, little_label = gen_save_data(jumping, player, cacti)
    data.append(little_data)
    labels.append(little_label)
    last_cacti += 1
    time += 5

    pygame.display.update()
    clock.tick(60)
  return False


def gen_data_and_play_game(model):
  game_over = False
  player = Player()
  cacti = []
  last_cacti = 0
  time = 4000
  data = []  # this game data
  labels = []
  while not game_over:
    if random.randint(0, 30) == 0 and last_cacti > 100:
      last_cacti = 0
      cacti.append(Cactus())
    if player.hit_cacti(cacti):
      break
    jumping = False
    jumping_data = model.predict(gen_live_data(cacti, player))[0]
    if jumping_data[0] > jumping_data[1]:
      player.jump()
      jumping = True

    player.update()

    i = 0
    while i < len(cacti):
      cactus = cacti[i]
      if cactus.update(time):  # returns is_ded (ded if offscreen)
        del (cacti[i])
        continue
      i += 1

    little_data, little_label = gen_save_data(jumping, player, cacti)
    data.append(little_data)
    labels.append(little_label)
    last_cacti += 1
    time += 5

  return data, labels


def disablePrint():
  sys.stdout = open(os.devnull, 'w')

def enablePrint():
  sys.stdout = sys.__stdout__


def main():
  pygame.init()
  pygame.font.init()

  font = pygame.font.SysFont('Comic Sans MS', 200)
  game_over_surface = font.render('Game Over', True, (255, 128, 128))
  font = pygame.font.SysFont('Comic Sans MS', 100)
  game_over_surface_2 = font.render('Press any key to restart', True, (255, 128, 128))

  screen = pygame.display.set_mode((800, 600))
  pygame.display.set_caption('Dino Dude')

  clock = pygame.time.Clock()

  play_game(None, screen)


if __name__ == '__main__':
  mainAI()
