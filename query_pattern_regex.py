query_pattern_regex = [
    '^select\s+ra,\s+dec,\s+z\s+from\s+specObj\s+where',
    '^select\s+G\.objID,\s*G\.ra,\s*G\.dec,\s*G\.u,\s*G\.g,\s*G\.r,\s*G\.i,\s*G\.z,\s*G\.psfMagErr_u\s+AS\s+u_err,\s*G\.psfMagErr_g\s+AS\s+g_err,\s*G\.psfMagErr_r\s+AS\s+r_err,\s*G\.psfMagErr_i\s+AS\s+i_err,\s*G\.psfMagErr_z\s+AS\s+z_err,\s*n\.distance,\s*P\.pId,\s*P\.version,\s*P\.z,\s*P\.zErr,\s*P\.t,\s*P\.tErr,\s*P\.quality,\s*S\.SpecObjId,\s*S\.ra,\s*S\.dec,\s*S\.z\s+from\s+Galaxy\s+as\s+G\s+left\s+outer\s+join\s+SpecObjAll\s+S\s+on\s+G\.objID=S\.bestObjID,\s*Photoz\s+as\s+P,\s*dbo\.fGetNearbyObjEq\(',
    '^select\s+(?:top\s+\d+\s+)?p\.objID,\s*p\.run,\s*p\.rerun,\s*p\.camcol,\s*p\.field,\s*p\.obj,\s*p\.type,\s*p\.ra,\s*p\.dec,\s*p\.u,\s*p\.g,p\.r,\s*p\.i,p\.z,\s*p\.Err_u,\s*p\.Err_g,\s*p\.Err_r,\s*p\.Err_i,\s*p\.Err_z\s+from\s+fGetNearbyObjEq\((:?\+|-)?(:?\d+)\.(:?\d+),(:?\+|-)?(:?\d+).(:?\d+),(:?\+|-)?(?:(:?\d+)\.(:?\d+)|(:?\d+))\)\s+n,\s*PhotoPrimary\s+p\s+where\s+n\.objID=p\.objID',
    '^select z,mag_1 from SpecObj where tile=\d+ and \(primTarget>=64 and primTarget<128\)',
    '^select\s+\*\s+from\s+PhotoZ\s+where\s+objId=0x(?:\d|\w){16}\s*',
    '^select\s+\*\s+from\s+Field\s+where\s+fieldId=0x(?:\d|\w){16}\s*',
    '^select\s+\*\s+from\s+SpecLine\s+where\s+specObjId=0x(?:\d|\w){16}\s*',
    '^select\s+\*\s+from\s+SpecLineIndex\s+where\s+specObjId=0x(?:\d|\w){16}\s*',
    '^select\s+\*\s+from\s+XCRedshift\s+where\s+specObjId=0x(?:\d|\w){16}\s*',
    '^select\s+\*\s+from\s+ELRedshift\s+where\s+specObjId=0x(?:\d|\w){16}\s*',
    '^select BestObjID, [S|L]\.SpecObjID, L\.lineID, L\.ew, L\.nSigma from SpecObj as S, SpecLine as L,  SpecLineNames as LN  where S\.SpecObjID = L\.SpecObjID  and S\.sn_1 > 10 and S\.zWarning = 0 and S\.eClass > -0\.1  group by (?:L\.SpecObjID|S\.BestObjID) order by (?:L\.SpecObjID|S\.BestObjID)',
    '^select \* from dbo\.(?:fHTMCover|fHTMCovehnjjr)\((?:\'|\")CIRCLE CARTESIAN 1 0 0 1(?:\'|\")\)  ',
]
