function cmdout = git_current_branch()
  cmd = "git branch | grep \* | cut -d ' ' -f2";
  [status, cmdout] = system(cmd);
  cmdout = strip(cmdout);
end
