"""Lossless bounded JSON envelope. Physical bytes remain the artifact identity.

The historical inventory names remain unchanged. A frozen encoding declaration,
magic, raw length and SHA256 distinguish compressed JSON from ordinary metadata.
Inference journals seal only after runtime close. Their exact raw bytes are
materialized for the unchanged accepted streaming verifier.
"""
import gzip
import hashlib
import json
import struct
import zlib
import os
from pathlib import Path

MAGIC = b'SPLJZ01\n'
MAX_RAW = 128 * 1024**2
HEADER = len(MAGIC) + 8 + 32
JOURNAL_MAGIC = b'SPLJL01\n'
MAX_JOURNAL_RAW = 4 * 1024**3
COMPRESSED_DIRECTORIES = frozenset({'moves', 'raw_results', 'service_calls', 'captures', 'targets'})


def encode_bytes(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_RAW:
        raise ValueError('Bounded nonempty original JSON bytes required')
    return MAGIC + struct.pack('<Q', len(raw)) + hashlib.sha256(raw).digest() + gzip.compress(raw, compresslevel=1, mtime=0)


def decode_bytes(stored):
    if type(stored) is not bytes or len(stored) > MAX_RAW + 1024**2:
        raise ValueError('Bounded physical evidence bytes required')
    if not stored.startswith(MAGIC):
        if stored.startswith(b'SPLJ') or stored.startswith(b'\x1f\x8b'):
            raise ValueError('Unknown or unbound compressed evidence')
        if not 0 < len(stored) <= MAX_RAW:
            raise ValueError('Bounded plain JSON bytes required')
        return stored
    if len(stored) <= HEADER:
        raise ValueError('Truncated compression header')
    size = struct.unpack('<Q', stored[len(MAGIC):len(MAGIC)+8])[0]
    if not 0 < size <= MAX_RAW:
        raise ValueError('Invalid declared raw evidence size')
    inflater = zlib.decompressobj(31)
    raw = inflater.decompress(stored[HEADER:], size+1)
    if not inflater.eof or inflater.unused_data or inflater.unconsumed_tail or len(raw) != size:
        raise ValueError('Truncated, trailing, concatenated or overlong compressed evidence')
    if hashlib.sha256(raw).digest() != stored[len(MAGIC)+8:HEADER]:
        raise ValueError('Raw evidence digest mismatch')
    return raw


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def parse_bytes(stored):
    return json.loads(decode_bytes(stored).decode('utf-8'), object_pairs_hook=unique_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))


def contract():
    return dict(schema='splendor_lossless_json_envelope_v1', magic_hex=MAGIC.hex(),
        encoding='raw_length_u64le_sha256_gzip_single_member', gzip_level=1,
        maximum_uncompressed_bytes=MAX_RAW, original_inventory_names_preserved=True,
        compressed_directories=sorted(COMPRESSED_DIRECTORIES), inference_journals='sealed_only_after_runtime_close',
        journal_magic_hex=JOURNAL_MAGIC.hex(),maximum_raw_journal_bytes=MAX_JOURNAL_RAW,
        journal_reader='strict_single_gzip_member_to_temporary_original_bytes_then_unchanged_scanner',
        hashes_bind='physical_stored_bytes; envelope additionally authenticates exact original bytes',
        no_precision_change=True, no_dropped_evidence=True)


def stream_journal(source, sink=None):
    """Authenticate a single bounded member, including raw length/hash and CRC."""
    path=Path(source);before=path.stat();raw_hash=hashlib.sha256();stored_hash=hashlib.sha256();total=0
    with path.open('rb') as stream:
        header=stream.read(HEADER);stored_hash.update(header)
        if len(header)!=HEADER or not header.startswith(JOURNAL_MAGIC):raise ValueError('Journal seal header')
        declared=struct.unpack('<Q',header[len(JOURNAL_MAGIC):len(JOURNAL_MAGIC)+8])[0]
        if not 0 < declared <= MAX_JOURNAL_RAW:raise ValueError('Bounded raw journal length')
        inflater=zlib.decompressobj(31)
        while True:
            block=stream.read(1024**2)
            if not block:break
            stored_hash.update(block)
            if inflater.eof:raise ValueError('Trailing journal bytes or extra gzip member')
            pending=block
            while pending:
                raw=inflater.decompress(pending,1024**2);pending=inflater.unconsumed_tail
                total+=len(raw)
                if total>declared:raise ValueError('Journal exceeds declared raw size')
                raw_hash.update(raw)
                if sink is not None:sink.write(raw)
                if inflater.unused_data:raise ValueError('Trailing journal bytes or extra gzip member')
        if not inflater.eof or total!=declared or raw_hash.digest()!=header[len(JOURNAL_MAGIC)+8:HEADER]:
            raise ValueError('Truncated journal, incorrect raw hash or size')
    after=path.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('Journal changed during decoding')
    return dict(raw_bytes=total,raw_sha256=raw_hash.hexdigest(),stored_bytes=after.st_size,stored_sha256=stored_hash.hexdigest())


def seal_journal(path):
    """Commit exact compressed bytes only after durable independent roundtrip.

    Called after every native/inference operation is closed. On precommit error,
    the original raw journal and any partial output remain. No historical path is
    accepted. The receipt is durable before replacing the new runtime journal.
    """
    from . import common as c
    path=Path(path).resolve()
    c.require(path.is_relative_to(c.OUT.resolve()) and path.name=='inference.jsonl','Only a newly closed inference journal')
    before=path.stat();temporary=path.with_name(path.name+'.sealing');hasher=hashlib.sha256();total=0
    with path.open('rb') as source,temporary.open('xb') as dest:
        dest.write(b'\0'*HEADER);compressor=zlib.compressobj(1,zlib.DEFLATED,31)
        for block in iter(lambda:source.read(1024**2),b''):
            total+=len(block);c.require(total<=MAX_JOURNAL_RAW,'Actual bounded journal scratch requirement')
            hasher.update(block);dest.write(compressor.compress(block))
        dest.write(compressor.flush());dest.seek(0)
        dest.write(JOURNAL_MAGIC+struct.pack('<Q',total)+hasher.digest());dest.flush();os.fsync(dest.fileno())
    after=path.stat();c.require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'Closed raw journal unchanged during sealing')
    checked=stream_journal(temporary)
    c.require(checked['raw_bytes']==total==after.st_size and checked['raw_sha256']==hasher.hexdigest(),'Exact original byte roundtrip before commit')
    raw=c.artifact(path);target=dict(path=raw['path'],bytes=checked['stored_bytes'],sha256=checked['stored_sha256'])
    receipt=dict(schema='closed_journal_lossless_seal_v1',raw=raw,stored=target,roundtrip=checked,
        all_operations_closed=True,physical_encoding='SPLJL01_single_gzip_member',no_scientific_bytes_removed=True)
    c.save(path.with_name(path.name+'.seal_receipt.json'),receipt)
    os.replace(temporary,path)
    c.require(c.artifact(path)==target,'Durable sealed physical journal identity')
    return c.save(path.with_name(path.name+'.seal_commit.json'),dict(status='committed',
        receipt=c.artifact(path.with_name(path.name+'.seal_receipt.json')),stored=target))
