### Overview

This section allows configuration of the "Recent Windows" feature in Niri.

The available settings and their respective default values are summarized below:

```kdl
recent-windows {
    // off

    binds {
        Alt+Tab         { next-window; }
        Alt+Shift+Tab   { previous-window; }
        Alt+grave       { next-window filter="app-id"; }
        Alt+Shift+grave { previous-window filter="app-id"; }

        Mod+Tab         { next-window; }
        Mod+Shift+Tab   { previous-window; }
        Mod+grave       { next-window filter="app-id"; }
        Mod+Shift+grave { previous-window filter="app-id"; }
    }
}
```

TODO hardcoded binds when open

### `off`

`off` disables the Recent Windows interface.

### `binds`

TODO
- have preference over the normal binds
- must have modifier
- next-window, previous-window, scope, filter
- having binds section anywhere in the config removes default recent-windows binds
