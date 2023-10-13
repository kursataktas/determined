import { ConfigProvider, theme as AntdTheme } from 'antd';
import { ThemeConfig } from 'antd/es/config-provider/context';
import React, {
  Dispatch,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from 'react';
import userSettings from 'stores/userSettings';
import { BrandingType, RecordKey } from 'components/kit/internal/types';

import { themes } from './themes';
import { DarkLight, getCssVar, globalCssVars, Mode, Theme } from './themeUtils';

interface StateUI {
  chromeCollapsed: boolean;
  darkLight: DarkLight;
  isPageHidden: boolean;
  mode: Mode;
  showChrome: boolean;
  showSpinner: boolean;
  theme: Theme;
}

const initUI: StateUI = {
  chromeCollapsed: false,
  darkLight: DarkLight.Light,
  isPageHidden: false,
  mode: Mode.System,
  showChrome: true,
  showSpinner: false,
  theme: {} as Theme,
};

const StoreActionUI = {
  HideUIChrome: 'HideUIChrome',
  HideUISpinner: 'HideUISpinner',
  SetMode: 'SetMode',
  SetPageVisibility: 'SetPageVisibility',
  SetTheme: 'SetTheme',
  ShowUIChrome: 'ShowUIChrome',
  ShowUISpinner: 'ShowUISpinner',
} as const;

type ActionUI =
  | { type: typeof StoreActionUI.HideUIChrome }
  | { type: typeof StoreActionUI.HideUISpinner }
  | { type: typeof StoreActionUI.SetMode; value: Mode }
  | { type: typeof StoreActionUI.SetPageVisibility; value: boolean }
  | { type: typeof StoreActionUI.SetTheme; value: { darkLight: DarkLight; theme: Theme } }
  | { type: typeof StoreActionUI.ShowUIChrome }
  | { type: typeof StoreActionUI.ShowUISpinner };

class UIActions {
  constructor(private dispatch: Dispatch<ActionUI>) { }

  public hideChrome = (): void => {
    this.dispatch({ type: StoreActionUI.HideUIChrome });
  };

  public hideSpinner = (): void => {
    this.dispatch({ type: StoreActionUI.HideUISpinner });
  };

  public setMode = (mode: Mode): void => {
    this.dispatch({ type: StoreActionUI.SetMode, value: mode });
  };

  public setPageVisibility = (isPageHidden: boolean): void => {
    this.dispatch({ type: StoreActionUI.SetPageVisibility, value: isPageHidden });
  };

  public setTheme = (darkLight: DarkLight, theme: Theme): void => {
    this.dispatch({ type: StoreActionUI.SetTheme, value: { darkLight, theme } });
  };

  public showChrome = (): void => {
    this.dispatch({ type: StoreActionUI.ShowUIChrome });
  };

  public showSpinner = (): void => {
    this.dispatch({ type: StoreActionUI.ShowUISpinner });
  };
}

const MATCH_MEDIA_SCHEME_DARK = '(prefers-color-scheme: dark)';
const MATCH_MEDIA_SCHEME_LIGHT = '(prefers-color-scheme: light)';
const ANTD_THEMES: Record<DarkLight, ThemeConfig> = {
  [DarkLight.Dark]: {
    algorithm: AntdTheme.darkAlgorithm,
    components: {
      Button: {
        colorBgContainer: 'transparent',
      },
      Checkbox: {
        colorBgContainer: 'transparent',
      },
      DatePicker: {
        colorBgContainer: 'transparent',
      },
      Input: {
        colorBgContainer: 'transparent',
      },
      InputNumber: {
        colorBgContainer: 'transparent',
      },
      Modal: {
        colorBgElevated: 'var(--theme-stage)',
      },
      Pagination: {
        colorBgContainer: 'transparent',
      },
      Progress: {
        marginXS: 0,
      },
      Radio: {
        colorBgContainer: 'transparent',
      },
      Select: {
        colorBgContainer: 'transparent',
      },
      Tree: {
        colorBgContainer: 'transparent',
      },
    },
    token: {
      borderRadius: 2,
      colorLink: '#57a3fa',
      colorLinkHover: '#8dc0fb',
      colorPrimary: '#1890ff',
      fontFamily: 'var(--theme-font-family)',
    },
  },
  [DarkLight.Light]: {
    algorithm: AntdTheme.defaultAlgorithm,
    components: {
      Button: {
        colorBgContainer: 'transparent',
      },
      Progress: {
        marginXS: 0,
      },
      Tooltip: {
        colorBgDefault: 'var(--theme-float)',
        colorTextLightSolid: 'var(--theme-float-on)',
      },
    },
    token: {
      borderRadius: 2,
      colorPrimary: '#1890ff',
      fontFamily: 'var(--theme-font-family)',
    },
  },
};

const getDarkLight = (mode: Mode, systemMode: Mode): DarkLight => {
  const resolvedMode =
    mode === Mode.System ? (systemMode === Mode.System ? Mode.Light : systemMode) : mode;
  return resolvedMode === Mode.Light ? DarkLight.Light : DarkLight.Dark;
};

const getSystemMode = (): Mode => {
  const isDark = matchMedia?.(MATCH_MEDIA_SCHEME_DARK).matches;
  if (isDark) return Mode.Dark;

  const isLight = matchMedia?.(MATCH_MEDIA_SCHEME_LIGHT).matches;
  if (isLight) return Mode.Light;

  return Mode.System;
};

const camelCaseToKebab = (text: string): string => {
  return text
    .trim()
    .split('')
    .map((char, index) => {
      return char === char.toUpperCase() ? `${index !== 0 ? '-' : ''}${char.toLowerCase()}` : char;
    })
    .join('');
};

/**
 * return a part of the input state that should be updated.
 * @param state ui state
 * @param action
 * @returns
 */
const reducerUI = (state: StateUI, action: ActionUI): Partial<StateUI> | void => {
  switch (action.type) {
    case StoreActionUI.HideUIChrome:
      if (!state.showChrome) return;
      return { showChrome: false };
    case StoreActionUI.HideUISpinner:
      if (!state.showSpinner) return;
      return { showSpinner: false };
    case StoreActionUI.SetMode:
      return { mode: action.value };
    case StoreActionUI.SetPageVisibility:
      return { isPageHidden: action.value };
    case StoreActionUI.SetTheme:
      return {
        darkLight: action.value.darkLight,
        theme: action.value.theme,
      };
    case StoreActionUI.ShowUIChrome:
      if (state.showChrome) return;
      return { showChrome: true };
    case StoreActionUI.ShowUISpinner:
      if (state.showSpinner) return;
      return { showSpinner: true };
    default:
      return;
  }
};
const StateContext = React.createContext<StateUI | undefined>(undefined);
const DispatchContext = React.createContext<Dispatch<ActionUI> | undefined>(undefined);

const reducer = (state: StateUI, action: ActionUI): StateUI => {
  const newState = reducerUI(state, action);
  return { ...state, ...newState }; // TODO: check for deep equality here instead of on the full state
};

const useUI = (): { actions: UIActions; ui: StateUI } => {
  const context = useContext(StateContext);
  if (context === undefined) {
    throw new Error('useStore(UI) must be used within a UIProvider');
  }
  const dispatchContext = useContext(DispatchContext);
  if (dispatchContext === undefined) {
    throw new Error('useStoreDispatch must be used within a UIProvider');
  }
  const uiActions = useMemo(() => new UIActions(dispatchContext), [dispatchContext]);
  return { actions: uiActions, ui: context };
};

export const ThemeHandler: React.FC<{
  children?: React.ReactNode;
  lightTheme: Theme;
  darkTheme: Theme;
}> = ({ children, lightTheme, darkTheme }) => {
  const [state, dispatch] = useReducer(reducer, initUI);
  // const systemMode = getSystemMode();
  // const userTheme = userSettings.get
  const theme = state.mode === DarkLight.Dark ? darkTheme : lightTheme;

  // Update darkLight and theme when branding, system mode, or mode changes.
  useLayoutEffect(() => {
    const darkLight = DarkLight.Light;
    dispatch({
      type: StoreActionUI.SetTheme,
      value: { darkLight, theme },
    });
  }, [state.mode]);

  return (
    <StateContext.Provider value={state}>
      <DispatchContext.Provider value={dispatch}>{children}</DispatchContext.Provider>
    </StateContext.Provider>
  );
};

export const UIProvider: React.FC<{
  children?: React.ReactNode;
  theme: Theme;
}> = ({ children, theme }) => {
  const ref = useRef<HTMLDivElement>(null);

  // Set global CSS variables shared across themes.
  Object.keys(globalCssVars).forEach((key) => {
    const value = (globalCssVars as Record<RecordKey, string>)[key];
    ref.current?.style.setProperty(`--${camelCaseToKebab(key)}`, value);
  });

  // Set each theme property as top level CSS variable.
  Object.keys(theme).forEach((key) => {
    const value = (theme as Record<RecordKey, string>)[key];
    ref.current?.style.setProperty(`--theme-${camelCaseToKebab(key)}`, value);
  });

  const configTheme = {
    algorithm: AntdTheme.defaultAlgorithm,
    components: {
      Button: {
        colorBgContainer: 'transparent',
      },
      Progress: {
        marginXS: 0,
      },
      Tooltip: {
        colorBgDefault: 'var(--theme-float)',
        colorTextLightSolid: 'var(--theme-float-on)',
      },
    },
    token: {
      borderRadius: 2,
      colorPrimary: '#1890ff',
      fontFamily: 'var(--theme-font-family)',
    },
  };

  return (
    <div className="ui-provider" ref={ref}>
      <ConfigProvider theme={configTheme}>{children}</ConfigProvider>
    </div>
  );
};

export default useUI;
