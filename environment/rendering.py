"""
rendering.py - Sista Health WhatsApp-Style Visualization
==========================================================
Renders a WhatsApp-style chat interface showing the RL agent
making response decisions for Nigerian women's health queries.
"""

import numpy as np

WA_BG           = (236, 229, 221)
WA_HEADER       = (7,   94,  84)
WA_BUBBLE_USER  = (255, 255, 255)
WA_BUBBLE_AGENT = (220, 248, 198)
WA_SIDEBAR      = (240, 240, 240)
WA_GREEN        = (37,  211, 102)
WA_DARK_TEXT    = (17,  17,  17)
WA_GRAY_TEXT    = (102, 102, 102)
WA_TIME_TEXT    = (144, 144, 144)
WA_TICK         = (83,  175, 236)
WA_RED          = (220, 53,  69)
WA_ORANGE       = (255, 165, 0)
WA_BLUE         = (52,  144, 220)
WA_ONLINE       = (77,  182, 172)

ACTION_COLORS = {
    0: WA_BLUE,
    1: WA_GREEN,
    2: WA_RED,
    3: WA_ORANGE,
}

ACTION_ICONS_TEXT = {
    0: "[TXT]",
    1: "[VOI]",
    2: "[SOS]",
    3: "[?]",
}

WINDOW_W = 980
WINDOW_H = 650


def draw_rounded_rect(surface, color, rect, radius):
    import pygame
    x, y, w, h = rect
    pygame.draw.rect(surface, color, (x + radius, y, w - 2*radius, h))
    pygame.draw.rect(surface, color, (x, y + radius, w, h - 2*radius))
    pygame.draw.circle(surface, color, (x + radius,     y + radius),     radius)
    pygame.draw.circle(surface, color, (x + w - radius, y + radius),     radius)
    pygame.draw.circle(surface, color, (x + radius,     y + h - radius), radius)
    pygame.draw.circle(surface, color, (x + w - radius, y + h - radius), radius)


def render_frame(env):
    import pygame

    if env.screen is None:
        pygame.init()
        pygame.display.init()
        env.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Sista Health - RL Agent")
        env.clock = pygame.time.Clock()
        if not hasattr(env, '_chat_history'):
            env._chat_history = []

    try:
        font_title  = pygame.font.SysFont("Arial", 17, bold=True)
        font_medium = pygame.font.SysFont("Arial", 14)
        font_small  = pygame.font.SysFont("Arial", 12)
        font_bold   = pygame.font.SysFont("Arial", 13, bold=True)
        font_large  = pygame.font.SysFont("Arial", 20, bold=True)
    except Exception:
        font_title  = pygame.font.Font(None, 20)
        font_medium = pygame.font.Font(None, 17)
        font_small  = pygame.font.Font(None, 15)
        font_bold   = pygame.font.Font(None, 16)
        font_large  = pygame.font.Font(None, 24)

    surface = env.screen
    surface.fill(WA_BG)

    # Left sidebar
    SIDEBAR_W = 280
    pygame.draw.rect(surface, WA_SIDEBAR, (0, 0, SIDEBAR_W, WINDOW_H))
    pygame.draw.line(surface, (200, 200, 200), (SIDEBAR_W, 0), (SIDEBAR_W, WINDOW_H), 1)

    # Sidebar header
    pygame.draw.rect(surface, WA_HEADER, (0, 0, SIDEBAR_W, 58))
    title_surf = font_large.render("Sista Health", True, (255, 255, 255))
    surface.blit(title_surf, (16, 18))

    # Search bar
    pygame.draw.rect(surface, (255, 255, 255), (8, 65, SIDEBAR_W - 16, 36), border_radius=18)
    search_surf = font_small.render("Search conversations...", True, WA_GRAY_TEXT)
    surface.blit(search_surf, (24, 75))

    # Contact list
    contacts = [
        ("Amara (Pidgin)",     "Maternal Health",  WA_GREEN,  True),
        ("Chidinma (English)", "Sexual Health",     WA_BLUE,   False),
        ("Folake (Yoruba)",    "Antenatal Care",    WA_ORANGE, False),
        ("Ngozi (English)",    "Family Planning",   WA_GREEN,  False),
        ("Aisha (Hausa)",      "Postpartum Care",   WA_BLUE,   False),
    ]

    y_contact = 112
    for i, (name, topic, color, active) in enumerate(contacts):
        bg = (220, 248, 198) if active else WA_SIDEBAR
        pygame.draw.rect(surface, bg, (0, y_contact, SIDEBAR_W, 52))
        if active:
            pygame.draw.rect(surface, WA_GREEN, (0, y_contact, 4, 52))
        pygame.draw.circle(surface, color, (30, y_contact + 26), 20)
        init_surf  = font_bold.render(name[0], True, (255, 255, 255))
        name_surf  = font_bold.render(name,    True, WA_DARK_TEXT)
        topic_surf = font_small.render(topic,  True, WA_GRAY_TEXT)
        surface.blit(init_surf,  (24, y_contact + 17))
        surface.blit(name_surf,  (60, y_contact + 10))
        surface.blit(topic_surf, (60, y_contact + 30))
        pygame.draw.line(surface, (220, 220, 220),
                         (60, y_contact + 52), (SIDEBAR_W - 8, y_contact + 52), 1)
        y_contact += 52

    # Chat area
    CHAT_X = SIDEBAR_W + 1
    CHAT_W  = WINDOW_W - SIDEBAR_W

    # Chat header
    pygame.draw.rect(surface, WA_HEADER, (CHAT_X, 0, CHAT_W, 58))

    if env.state is not None:
        info = env._get_info()
        pygame.draw.circle(surface, WA_GREEN, (CHAT_X + 26, 29), 20)
        av_surf     = font_large.render("S", True, (255, 255, 255))
        name_surf   = font_title.render(
            f"Sista Health  ({info['language']})", True, (255, 255, 255))
        status_surf = font_small.render(
            f"online  |  {info['domain']}  |  Step {info['step']}/9",
            True, (160, 220, 180))
        surface.blit(av_surf,     (CHAT_X + 19, 17))
        surface.blit(name_surf,   (CHAT_X + 56, 12))
        surface.blit(status_surf, (CHAT_X + 56, 34))
        pygame.draw.circle(surface, WA_ONLINE, (CHAT_X + 44, 44), 5)
    else:
        name_surf = font_title.render("Sista Health", True, (255, 255, 255))
        surface.blit(name_surf, (CHAT_X + 56, 18))

    # Chat messages area
    chat_area_y = 65
    chat_area_h = WINDOW_H - 65 - 120
    pygame.draw.rect(surface, WA_BG, (CHAT_X, chat_area_y, CHAT_W, chat_area_h))

    # Wallpaper dots
    for dot_y in range(chat_area_y + 15, chat_area_y + chat_area_h, 30):
        for dot_x in range(CHAT_X + 15, CHAT_X + CHAT_W, 30):
            pygame.draw.circle(surface, (220, 213, 206), (dot_x, dot_y), 1)

    # Add message to history
    if env.last_action is not None and env.state is not None:
        info = env._get_info()
        msg = {
            "step":        int(env.state[5]),
            "action":      env.last_action,
            "action_name": env.ACTIONS[env.last_action],
            "reward":      env.last_reward,
            "feedback":    env.last_feedback,
            "language":    info["language"],
            "urgency":     info["urgency"],
            "literacy":    info["literacy"],
            "topic":       info["topic"],
        }
        if not hasattr(env, '_chat_history'):
            env._chat_history = []
        if len(env._chat_history) == 0 or \
           env._chat_history[-1]["step"] != msg["step"]:
            env._chat_history.append(msg)

    # Draw messages
    if hasattr(env, '_chat_history') and env._chat_history:
        visible = env._chat_history[-5:]
        msg_y   = chat_area_y + 12

        for msg in visible:
            action_color = ACTION_COLORS[msg["action"]]
            reward_sign  = "+" if msg["reward"] >= 0 else ""

            # User bubble (left)
            user_text = f"{msg['language']} user  |  {msg['topic'][:22]}  |  {msg['urgency']}"
            user_surf = font_small.render(user_text, True, WA_DARK_TEXT)
            user_w    = user_surf.get_width() + 24
            user_h    = 36
            draw_rounded_rect(surface, WA_BUBBLE_USER,
                               (CHAT_X + 12, msg_y, user_w, user_h), 8)
            pygame.draw.rect(surface, (200, 200, 200),
                             (CHAT_X + 12, msg_y, user_w, user_h), 1, border_radius=8)
            surface.blit(user_surf, (CHAT_X + 24, msg_y + 11))
            time_surf = font_small.render(f"Step {msg['step']}", True, WA_TIME_TEXT)
            surface.blit(time_surf, (CHAT_X + 24, msg_y + user_h + 2))

            # Agent bubble (right)
            agent_text  = f"{ACTION_ICONS_TEXT[msg['action']]}  {msg['action_name']}"
            reward_text = f"{reward_sign}{msg['reward']:.0f} pts"
            agent_surf  = font_bold.render(agent_text, True, action_color)
            reward_surf = font_small.render(
                reward_text, True,
                (34, 139, 34) if msg["reward"] >= 0 else WA_RED)
            agent_w = max(agent_surf.get_width(),
                          reward_surf.get_width()) + 32
            agent_h = 52
            agent_x = CHAT_X + CHAT_W - agent_w - 16
            draw_rounded_rect(surface, WA_BUBBLE_AGENT,
                               (agent_x, msg_y, agent_w, agent_h), 8)
            surface.blit(agent_surf,  (agent_x + 12, msg_y + 8))
            surface.blit(reward_surf, (agent_x + 12, msg_y + 28))
            tick_surf = font_small.render("vv", True, WA_TICK)
            surface.blit(tick_surf, (agent_x + agent_w - 28, msg_y + agent_h - 16))

            msg_y += 68

    # Input bar
    input_y = WINDOW_H - 120
    pygame.draw.rect(surface, (240, 240, 240), (CHAT_X, input_y, CHAT_W, 1))
    pygame.draw.rect(surface, (249, 249, 249), (CHAT_X, input_y, CHAT_W, 54))
    pygame.draw.rect(surface, (255, 255, 255),
                     (CHAT_X + 12, input_y + 10, CHAT_W - 80, 34), border_radius=17)

    if env.state is not None and env.last_feedback:
        hint = env.last_feedback[:60] + "..." \
               if len(env.last_feedback) > 60 else env.last_feedback
    else:
        hint = "Waiting for user message..."
    hint_surf = font_small.render(hint, True, WA_GRAY_TEXT)
    surface.blit(hint_surf, (CHAT_X + 26, input_y + 20))

    pygame.draw.circle(surface, WA_GREEN,
                       (CHAT_X + CHAT_W - 30, input_y + 27), 18)
    send_surf = font_bold.render(">", True, (255, 255, 255))
    surface.blit(send_surf, (CHAT_X + CHAT_W - 36, input_y + 18))

    # Stats bar
    stats_y = WINDOW_H - 66
    pygame.draw.rect(surface, WA_HEADER, (CHAT_X, stats_y, CHAT_W, 66))

    if env.state is not None:
        info     = env._get_info()
        ep_color = WA_GREEN if env.episode_reward >= 0 else WA_RED
        ep_label = font_small.render("Episode Reward", True, (160, 220, 180))
        ep_val   = font_large.render(f"{env.episode_reward:.0f}", True, ep_color)
        surface.blit(ep_label, (CHAT_X + 16, stats_y + 8))
        surface.blit(ep_val,   (CHAT_X + 16, stats_y + 26))

        # Step progress bar
        bar_x = CHAT_X + 130
        bar_y = stats_y + 24
        bar_w = 180
        bar_h = 10
        pygame.draw.rect(surface, (255, 255, 255),
                         (bar_x, bar_y, bar_w, bar_h), border_radius=5)
        fill_w = int(bar_w * (int(env.state[5]) / 10))
        if fill_w > 0:
            pygame.draw.rect(surface, WA_GREEN,
                             (bar_x, bar_y, fill_w, bar_h), border_radius=5)
        step_lbl = font_small.render(
            f"Step {int(env.state[5])}/10", True, (200, 230, 200))
        surface.blit(step_lbl, (bar_x, stats_y + 8))

        algo_surf = font_small.render("PPO Agent", True, (200, 230, 200))
        surface.blit(algo_surf, (CHAT_X + CHAT_W - 120, stats_y + 8))

        if env.last_action is not None:
            act_color = ACTION_COLORS[env.last_action]
            act_text  = env.ACTIONS[env.last_action]
            act_surf  = font_bold.render(f"Last: {act_text}", True, act_color)
            surface.blit(act_surf, (CHAT_X + CHAT_W - 160, stats_y + 30))

    pygame.event.pump()
    pygame.display.flip()
    env.clock.tick(env.metadata["render_fps"])

    return np.transpose(
        np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
    )
