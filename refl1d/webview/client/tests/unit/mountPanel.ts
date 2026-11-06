import { mount } from "@vue/test-utils";
import { type Component, defineComponent, h, KeepAlive } from "vue";
import { fakeSocket } from "./fakeSocket";

/**
 * Mount a panel with a fake socket answering from `responses`. Panels only
 * fetch and draw while active inside <KeepAlive>, as in PanelTabContainer.
 * Change entries of the returned `responses` to change what the server sends next.
 */
export function mountPanel(panel: Component, initialResponses: Record<string, unknown> = {}) {
  const responses: Record<string, unknown> = { get_shared_setting: "2026-01-01T00:00:00", ...initialResponses };
  const socket = fakeSocket(responses);
  const Host = defineComponent(() => () => h(KeepAlive, null, [h(panel, { socket })]));
  const wrapper = mount(Host, { attachTo: document.body });
  return { wrapper, socket, responses };
}
